# Work Days Calculator

This Prolog program calculates N working days from a given date in 2024, skipping weekends (Saturday and Sunday).

## Prerequisites

- SWI-Prolog must be installed on your system
- For macOS users, you can install it using Homebrew:
  ```bash
  brew install swi-prolog
  ```

## Project Structure

- `work_days.pl` - Main Prolog program
- `work_days_test.pl` - Test cases for the program
- `lab7_cg103_g10_v5_Ilci_Kara.pdf` - Project report

## Running the Program

### 1. Interactive Mode

To start the interactive Prolog shell with the program loaded:

```bash
swipl work_days.pl
```

Once in the interactive shell, you can run queries like:

```prolog
?- n_work_days("2205", 6).  % Calculate 6 working days from May 22, 2024
?- n_work_days("0106", 10). % Calculate 10 working days from June 1, 2024
```

### 2. Running Tests

To run all test cases and see their results:

```bash
swipl
```

Then in the Prolog shell, type:

```prolog
?- [work_days].
?- [work_days_test].
?- run_tests.
```

This will:

1. Load the main program (`work_days.pl`)
2. Load the test file (`work_days_test.pl`)
3. Run all tests and show you the results
4. You can type `halt.` to exit after seeing the results

## Input Format

The program takes two inputs:

1. Date string in "DDMM" format (e.g., "2205" for May 22)
2. Number of working days to add (must be ≤ 366)

## Output Format

The output will be in the format:

```
DayName, DDMM
```

For example:

```
Tuesday, 3005
```

## Example Queries

Here are some example queries you can try:

```prolog
% Calculate 6 working days from May 22, 2024
?- n_work_days("2205", 6).

% Calculate 10 working days from June 1, 2024
?- n_work_days("0106", 10).

% Calculate 0 working days (returns same date if it's a working day)
?- n_work_days("2205", 0).

% Invalid query (weekend start date)
?- n_work_days("2005", 1).  % May 20, 2024 is a Saturday
```

## Test Cases

The test file includes several test cases:

1. Basic examples
2. Edge case with zero working days
3. Invalid starting date (weekend)
4. Maximum days boundary test

## Notes

- The program only works with dates in 2024
- Starting dates must be working days (Monday-Friday)
- The number of working days to add must be ≤ 366
- The program automatically skips weekends (Saturday and Sunday)
