# Dataset Instructions

## `vocab.pkl`
1. It has an array
    - First element - array of all tokens in the robot language
    - Second Element - Dictionary matching token (of the robot language) to it's index - `token_to_index_array`

## `final_dataset.pkl`
- It's an array of data entries
- Each data entry has a **NL instruction, array of subset paths, BMs (Adjencency list corresponding to the NL instruction), Array of output behaviors for each subpaths (Ground Truth values)**

## `emb_matrix.pkl`
- In `.gitignore`
- It has GLoVE 100d vector for the natural language instruction's vocabulary