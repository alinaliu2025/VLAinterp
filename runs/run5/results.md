# run5: TDNV, image vs. instruction

- model: `openvla/openvla-7b-finetuned-libero-spatial`
- k = 10, N = 10 (crossed grid: Exp 1 and Exp 2 group the same hidden states)
- images: first frame of libero_spatial layout tasks [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], replicates = init states [0, 1, 2, 3, 4]
- variation: different tasks; seed 7; 200 label shuffles (init 0)
- hidden state: last prompt position (produces the first action token); readout 0 skipped (identical for all samples)

## Instruction set `suite`

0. pick up the black bowl between the plate and the ramekin and place it on the plate
1. pick up the black bowl next to the ramekin and place it on the plate
2. pick up the black bowl from table center and place it on the plate
3. pick up the black bowl on the cookie box and place it on the plate
4. pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate
5. pick up the black bowl on the ramekin and place it on the plate
6. pick up the black bowl next to the cookie box and place it on the plate
7. pick up the black bowl on the stove and place it on the plate
8. pick up the black bowl next to the plate and place it on the plate
9. pick up the black bowl on the wooden cabinet and place it on the plate

| features | Exp 1 (by image) min | Exp 2 (by instruction) min | Exp 1 / Exp 2 at last readout | curves cross | shuffled / real at each minimum |
|---|---|---|---|---|---|
| raw | 0.115 @ L3 | 0.147 @ L7 | 0.619 / 3.33 | between layers 5 and 6 | Exp 1 63x, Exp 2 36x |
| centered_unitnorm | 0.116 @ L3 | 0.151 @ L7 | 0.592 / 3.27 | between layers 5 and 6 | Exp 1 60x, Exp 2 35x |

## Instruction set `generic`

0. pick up the black bowl and place it on the plate
1. stack the two black bowls
2. turn on the stove
3. open the middle drawer of the wooden cabinet
4. open the bottom drawer of the wooden cabinet
5. open the bottom drawer and put the black bowl inside
6. push the plate to the front of the stove
7. move the plate next to the cookie box
8. push the cookie box to the front of the stove
9. pick up the plate and place it next to the ramekin

| features | Exp 1 (by image) min | Exp 2 (by instruction) min | Exp 1 / Exp 2 at last readout | curves cross | shuffled / real at each minimum |
|---|---|---|---|---|---|
| raw | 0.426 @ L3 | 0.0684 @ L7 | 1.02 / 3.06 | between layers 1 and 2 | Exp 1 15x, Exp 2 86x |
| centered_unitnorm | 0.422 @ L3 | 0.0675 @ L7 | 0.986 / 3.02 | between layers 1 and 2 | Exp 1 15x, Exp 2 83x |

## Notes (added by hand after the run)

- The "curves cross" column lists only the FIRST crossing. Both sets have three phases:
  layers 1-5 image-organized, 6-11 instruction-organized (minimum at L7), 12-32
  image-organized again but weakly. Suite crossings ~L6 and ~L11; generic ~L1.5, ~L5, ~L16.
- Shuffled / real is measured at each curve's own minimum, on init 0.
