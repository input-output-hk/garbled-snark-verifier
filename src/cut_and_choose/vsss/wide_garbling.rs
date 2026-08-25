use ark_ff::{BigInteger, PrimeField};
use ark_secp256k1::Fr;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::{EvaluatedWire, GarbledWire, S};

const TAG_LEN: usize = 16;

/// Domain-separation context for [`GarbledWideLabelTable::aggregate_hash`].
const AGGREGATE_HASH_CONTEXT: &str = "gsv_wide_lable_table_aggregate";

/// Reasons a `GarbledWideLabelTable` received from a counterparty is not a well-formed table.
///
/// An honest table has `2^k` rows of `k * 16 + TAG_LEN` bytes for some `k >= 1` — the shape
/// [`GarbledWideLabelTable::lookup_evaluated_wires_and_index`] relies on. Checking it at the
/// deserialization boundary is what lets that path stay total, rather than discovering a
/// malformed table part-way through an evaluation.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum InvalidWideLabelTable {
    #[error("table must have at least two rows, got {0}")]
    TooFewRows(usize),
    #[error("table row count must be a power of two, got {0}")]
    RowCountNotPowerOfTwo(usize),
    #[error("row {index} of a {rows}-row table has {actual} bytes, expected {expected}")]
    RowLength {
        index: usize,
        rows: usize,
        actual: usize,
        expected: usize,
    },
}

#[derive(Serialize, Deserialize, Hash, Debug, Clone)]
#[serde(try_from = "Vec<Vec<u8>>")]
pub struct GarbledWideLabelTable(Vec<Vec<u8>>);

impl TryFrom<Vec<Vec<u8>>> for GarbledWideLabelTable {
    type Error = InvalidWideLabelTable;

    fn try_from(rows: Vec<Vec<u8>>) -> Result<Self, Self::Error> {
        if rows.len() < 2 {
            return Err(InvalidWideLabelTable::TooFewRows(rows.len()));
        }
        if !rows.len().is_power_of_two() {
            return Err(InvalidWideLabelTable::RowCountNotPowerOfTwo(rows.len()));
        }

        let expected = rows.len().ilog2() as usize * 16 + TAG_LEN;
        for (index, row) in rows.iter().enumerate() {
            if row.len() != expected {
                return Err(InvalidWideLabelTable::RowLength {
                    index,
                    rows: rows.len(),
                    actual: row.len(),
                    expected,
                });
            }
        }

        Ok(GarbledWideLabelTable(rows))
    }
}

impl GarbledWideLabelTable {
    pub fn build_all(byte_labels: &[Fr], bit_labels: &[GarbledWire]) -> Vec<Self> {
        byte_labels
            .chunks(256)
            .zip(bit_labels.chunks(8))
            .map(|(byte_labels, bit_labels)| GarbledWideLabelTable::new(byte_labels, bit_labels))
            .collect()
    }

    /// Per-bit (width-1) tables: two value-labels per bit (one per polarity `β ∈ {0,1}`) and one
    /// `bit_label` per bit, producing one 2-entry table per bit. Used by the BABE
    /// `CheckLampAdaptorMatch` per-bit reveal, where each bit carries its own degree-`d` polynomial
    /// rather than the 8-bit byte groups of [`Self::build_all`].
    pub fn build_per_bit(bit_value_labels: &[Fr], bit_labels: &[GarbledWire]) -> Vec<Self> {
        bit_value_labels
            .chunks(2)
            .zip(bit_labels.iter())
            .map(|(vals, bit)| GarbledWideLabelTable::new(vals, std::slice::from_ref(bit)))
            .collect()
    }

    fn new(byte_labels: &[Fr], bit_labels: &[GarbledWire]) -> Self {
        assert_ne!(bit_labels.len(), 0);
        assert_ne!(byte_labels.len(), 0);
        assert_eq!(byte_labels.len(), 2usize.pow(bit_labels.len() as u32));

        let table = byte_labels
            .iter()
            .enumerate()
            .map(|(i, byte_label)| {
                let bit_labels: Vec<u8> = (0..bit_labels.len())
                    .map(|bit| {
                        if ((i >> (bit_labels.len() - bit - 1)) & 1) == 0 {
                            bit_labels[bit].label0
                        } else {
                            bit_labels[bit].label1
                        }
                    })
                    .flat_map(|label| label.to_bytes())
                    .chain(std::iter::repeat_n(0u8, TAG_LEN))
                    .collect();

                let mut blake_hash = blake3::Hasher::new();
                let mut mask = (0..bit_labels.len()).map(|_| 0u8).collect_vec();
                blake_hash.update(&byte_label.into_bigint().to_bytes_le());
                blake_hash.finalize_xof().fill(&mut mask);

                bit_labels
                    .iter()
                    .zip(mask.iter())
                    .map(|(c, m)| c ^ m)
                    .collect_vec()
            })
            .collect();
        GarbledWideLabelTable(table)
    }

    /// Hash over all the wide label lookup tables.
    ///
    /// The digest is length-framed (table count, per-table row count, per-row length) so that it
    /// binds the *partition* of the byte stream into tables and rows, not just the concatenated
    /// bytes. Without the framing a counterparty could deliver a differently-shaped
    /// `Vec<GarbledWideLabelTable>` with the same flattened bytes under the same commitment.
    /// It is also domain-separated so these digests do not share a namespace with any other
    /// bare-BLAKE3 commitment.
    pub fn aggregate_hash(tables: &[Self]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new_derive_key(AGGREGATE_HASH_CONTEXT);
        hasher.update(&(tables.len() as u64).to_le_bytes());
        for table in tables.iter() {
            hasher.update(&(table.0.len() as u64).to_le_bytes());
            for row in table.0.iter() {
                hasher.update(&(row.len() as u64).to_le_bytes());
                hasher.update(row);
            }
        }
        let hash = hasher.finalize();
        *hash.as_bytes()
    }

    pub fn lookup(&self, wide_label: &Fr) -> Vec<S> {
        self.lookup_evaluated_wires(wide_label)
            .iter()
            .map(|evaluated_wire| evaluated_wire.active_label)
            .collect_vec()
    }

    pub fn lookup_index(&self, wide_label: &Fr) -> usize {
        self.lookup_evaluated_wires_and_index(wide_label).0
    }

    pub fn lookup_evaluated_wires(&self, wide_label: &Fr) -> Vec<EvaluatedWire> {
        self.lookup_evaluated_wires_and_index(wide_label).1
    }

    pub fn lookup_evaluated_wires_and_index(&self, wide_label: &Fr) -> (usize, Vec<EvaluatedWire>) {
        self.try_lookup_evaluated_wires_and_index(wide_label)
            .expect("Failed to decrypt wide label lookup table with the given key")
    }

    /// Fallible form of [`Self::lookup_evaluated_wires_and_index`].
    ///
    /// Returns `None` when the table is malformed or when no row's tag clears under `wide_label`.
    /// The latter is the expected outcome for a well-formed but garbage-filled table from a
    /// counterparty: shape validation cannot catch it, because a wrong-keyed table is correctly
    /// shaped. Prefer this on any path handling tables received over the wire.
    pub fn try_lookup_evaluated_wires_and_index(
        &self,
        wide_label: &Fr,
    ) -> Option<(usize, Vec<EvaluatedWire>)> {
        // Guaranteed by `new` and by the `TryFrom` deserialization boundary, but re-checked here
        // so this method is total for every value of `Self` regardless of how it was built.
        let rows = self.0.len();
        if rows < 2 || !rows.is_power_of_two() {
            return None;
        }
        let label_count = rows.ilog2();
        let row_len = label_count as usize * 16 + TAG_LEN;
        if self.0.iter().any(|row| row.len() != row_len) {
            return None;
        }

        self.0
            .iter()
            .enumerate()
            .find_map(|(wide_label_idx, ciphertext)| {
                let mut mask = vec![0u8; ciphertext.len()];
                let mut blake_hash = blake3::Hasher::new();
                blake_hash.update(&wide_label.into_bigint().to_bytes_le());
                blake_hash.finalize_xof().fill(&mut mask[..]);

                let decrypted = ciphertext
                    .iter()
                    .zip(mask.iter())
                    .map(|(c, m)| c ^ m)
                    .collect_vec();

                let (labels, tag) = decrypted.split_at(decrypted.len() - TAG_LEN);

                let successful_decryption = tag.iter().all(|x| *x == 0);

                successful_decryption.then_some((
                    wide_label_idx,
                    labels
                        .chunks(16)
                        .enumerate()
                        .map(|(bit_idx, chunk)| {
                            let label = S::from_bytes(
                                chunk.try_into().expect("should be exactly 16 items"),
                            );
                            let bit_value =
                                ((wide_label_idx >> (label_count as usize - bit_idx - 1)) & 1) == 1;
                            EvaluatedWire::new(label, bit_value)
                        })
                        .collect_vec(),
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::UniformRand;
    use rand::thread_rng;

    use super::*;
    use crate::Delta;

    #[test]
    fn test_garbled_wide_label_table_lookup() {
        let mut rng = thread_rng();

        let delta = Delta::generate(&mut rng);

        for num_bits in [2, 8] {
            let num_labels = 2u32.pow(num_bits as u32);
            let bit_labels = (0..num_bits)
                .map(|_| GarbledWire::random(&mut rng, &delta))
                .collect_vec();
            let byte_labels = (0..num_labels).map(|_| Fr::rand(&mut rng)).collect_vec();

            let table = GarbledWideLabelTable::new(&byte_labels, &bit_labels);

            let expected_vals = (0..num_labels)
                .map(|i| {
                    (0..num_bits)
                        .map(|bit| {
                            if ((i >> (num_bits - bit - 1)) & 1) == 0 {
                                EvaluatedWire::new(bit_labels[bit].label0, false)
                            } else {
                                EvaluatedWire::new(bit_labels[bit].label1, true)
                            }
                        })
                        .collect_vec()
                })
                .collect_vec();

            for (byte_label, expected_vals) in byte_labels.iter().zip(expected_vals.iter()) {
                let recovered_bit_labels = table.lookup_evaluated_wires(byte_label);
                assert_eq!(&recovered_bit_labels[..], &expected_vals[..]);
            }

            println!("table size: {}", table.0.iter().flatten().count());
        }
    }

    fn sample_table(rng: &mut impl rand::Rng, num_bits: usize) -> (Vec<Fr>, GarbledWideLabelTable) {
        let delta = Delta::generate(rng);
        let bit_labels = (0..num_bits)
            .map(|_| GarbledWire::random(rng, &delta))
            .collect_vec();
        let byte_labels = (0..2usize.pow(num_bits as u32))
            .map(|_| Fr::rand(rng))
            .collect_vec();
        let table = GarbledWideLabelTable::new(&byte_labels, &bit_labels);
        (byte_labels, table)
    }

    #[test]
    fn test_deserialize_rejects_malformed_shape() {
        // Every one of these is a shape a counterparty could previously hand over and have
        // accepted, because `Deserialize` was derived straight onto the private `Vec<Vec<u8>>`.
        let malformed: Vec<Vec<Vec<u8>>> = vec![
            vec![],                             // empty: no rows at all
            vec![vec![0u8; 16]],                // k = 0, never built honestly
            vec![vec![0u8; 32]; 3],             // row count not a power of two
            vec![vec![0u8; 48], vec![0u8; 48]], // 2 rows, wrong row length
            vec![vec![0u8; 32], vec![0u8; 16]], // ragged rows
        ];

        for table in malformed {
            let json = serde_json::to_string(&table).unwrap();
            assert!(
                serde_json::from_str::<GarbledWideLabelTable>(&json).is_err(),
                "accepted malformed table with row lengths {:?}",
                table.iter().map(|r| r.len()).collect_vec(),
            );
        }
    }

    #[test]
    fn test_deserialize_accepts_honest_tables() {
        let mut rng = thread_rng();
        for num_bits in [1, 3, 8] {
            let (_, table) = sample_table(&mut rng, num_bits);
            let json = serde_json::to_string(&table).unwrap();
            let round_tripped: GarbledWideLabelTable = serde_json::from_str(&json).unwrap();
            assert_eq!(round_tripped.0, table.0);
        }
    }

    #[test]
    fn test_try_lookup_returns_none_instead_of_panicking() {
        let mut rng = thread_rng();
        let (byte_labels, table) = sample_table(&mut rng, 3);

        // Correctly shaped but wrong-keyed: no row's tag clears. Shape validation cannot catch
        // this, so the lookup path has to be able to say so without panicking.
        let wrong = Fr::rand(&mut rng);
        assert!(table.try_lookup_evaluated_wires_and_index(&wrong).is_none());

        let (idx, wires) = table
            .try_lookup_evaluated_wires_and_index(&byte_labels[5])
            .expect("honest label must decrypt");
        assert_eq!(idx, 5);
        assert_eq!(wires.len(), 3);
    }

    #[test]
    fn test_try_lookup_total_on_malformed_table() {
        // Constructed directly, bypassing `TryFrom` — `try_lookup` must still not panic.
        for rows in [vec![], vec![vec![0u8; 16]], vec![vec![0u8; 32]; 3]] {
            let table = GarbledWideLabelTable(rows);
            assert!(
                table
                    .try_lookup_evaluated_wires_and_index(&Fr::rand(&mut thread_rng()))
                    .is_none()
            );
        }
    }

    #[test]
    fn test_aggregate_hash_binds_partition() {
        // The same flattened byte stream, partitioned differently, must not collide.
        let row_a = vec![1u8; 32];
        let row_b = vec![2u8; 32];

        let two_rows = vec![GarbledWideLabelTable(vec![row_a.clone(), row_b.clone()])];
        let two_tables = vec![
            GarbledWideLabelTable(vec![row_a.clone()]),
            GarbledWideLabelTable(vec![row_b.clone()]),
        ];
        let one_row = vec![GarbledWideLabelTable(vec![
            row_a.iter().chain(row_b.iter()).copied().collect_vec(),
        ])];

        let h_two_rows = GarbledWideLabelTable::aggregate_hash(&two_rows);
        let h_two_tables = GarbledWideLabelTable::aggregate_hash(&two_tables);
        let h_one_row = GarbledWideLabelTable::aggregate_hash(&one_row);

        assert_ne!(h_two_rows, h_two_tables, "table partition must be bound");
        assert_ne!(h_two_rows, h_one_row, "row partition must be bound");
        assert_ne!(h_two_tables, h_one_row);
    }

    #[test]
    fn test_aggregate_hash_binds_per_bit_repartition() {
        // 11 honest per-bit tables (2 rows x 32 bytes = 64 bytes each) can be re-partitioned
        // into 8 single-row 16-byte tables plus 3 four-row 48-byte tables: identical flattened
        // bytes, identical table count, identical total length, and every table still satisfies
        // the `2^k` rows / `16k + TAG_LEN` row-length shape that `lookup_*` accepts. Only the
        // length framing in `aggregate_hash` separates the two.
        let stream = (0..64usize * 11).map(|i| i as u8).collect_vec();

        let partition = |shape: &[(usize, usize)]| {
            let mut rest = stream.as_slice();
            let tables = shape
                .iter()
                .map(|&(rows, row_len)| {
                    let table = (0..rows)
                        .map(|_| {
                            let (row, tail) = rest.split_at(row_len);
                            rest = tail;
                            row.to_vec()
                        })
                        .collect_vec();
                    GarbledWideLabelTable(table)
                })
                .collect_vec();
            assert!(rest.is_empty());
            tables
        };

        let honest = partition(&[(2, 32); 11]);
        let mut malicious_shape = vec![(1, 16); 8];
        malicious_shape.extend([(4, 48); 3]);
        let malicious = partition(&malicious_shape);

        assert_eq!(honest.len(), malicious.len());
        assert_ne!(
            GarbledWideLabelTable::aggregate_hash(&honest),
            GarbledWideLabelTable::aggregate_hash(&malicious),
        );
    }

    #[test]
    fn test_build_per_bit() {
        let mut rng = thread_rng();
        let delta = Delta::generate(&mut rng);
        let num_bits = 5;

        let bit_labels = (0..num_bits)
            .map(|_| GarbledWire::random(&mut rng, &delta))
            .collect_vec();
        // two value-labels per bit (β = 0, 1)
        let bit_value_labels = (0..num_bits * 2).map(|_| Fr::rand(&mut rng)).collect_vec();

        let tables = GarbledWideLabelTable::build_per_bit(&bit_value_labels, &bit_labels);
        assert_eq!(tables.len(), num_bits);

        for (bit, table) in tables.iter().enumerate() {
            for beta in 0..2usize {
                let wires = table.lookup_evaluated_wires(&bit_value_labels[bit * 2 + beta]);
                assert_eq!(wires.len(), 1);
                let expected = if beta == 0 {
                    EvaluatedWire::new(bit_labels[bit].label0, false)
                } else {
                    EvaluatedWire::new(bit_labels[bit].label1, true)
                };
                assert_eq!(wires[0], expected);
            }
        }
    }
}
