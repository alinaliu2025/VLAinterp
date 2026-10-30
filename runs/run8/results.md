# run5: TDNV, image vs. instruction

- model: `openvla/openvla-7b`
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
| raw | 0.369 @ L2 | 0.0345 @ L7 | 0.444 / 1.97 | between layers 1 and 2 | Exp 1 15x, Exp 2 1.9e+02x |
| centered_unitnorm | 0.352 @ L2 | 0.0362 @ L7 | 0.439 / 1.95 | between layers 1 and 2 | Exp 1 16x, Exp 2 1.7e+02x |

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
| raw | 1.28 @ L31 | 0.0104 @ L7 | 1.29 / 0.979 | no (exp2 lower at every layer) | Exp 1 3.7x, Exp 2 6.1e+02x |
| centered_unitnorm | 1.27 @ L31 | 0.0105 @ L7 | 1.28 / 0.968 | no (exp2 lower at every layer) | Exp 1 3.7x, Exp 2 5.9e+02x |

