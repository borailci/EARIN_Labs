% work_days_test.pl
% Unit tests for work_days.pl using SWI-Prolog's PlUnit

:- begin_tests(work_days).

% Code under test is loaded via swipl -l work_days.pl

% Helper to capture the printed output
capture_output(Goal, Output) :-
    with_output_to(string(Output), Goal).

% Test example cases
test(example1) :-
    capture_output(n_work_days("2205", 6), Out),
    assertion(Out == "Tuesday, 3005\n").

test(example2) :-
    capture_output(n_work_days("0106", 10), Out),
    assertion(Out == "Thursday, 1506\n").

% Edge case: zero working days (should return the same date if it's a working day)
test(zero_days) :-
    capture_output(n_work_days("2205", 0), Out),
    % May 22, 2024 is a Monday
    assertion(Out == "Monday, 2205\n").

% Invalid starting date (weekend) should fail
test(invalid_start, [fail]) :-
    n_work_days("2005", 1).  % May 20, 2024 is a Saturday

% Max N boundary
test(max_days) :-
    % Should succeed for N = 366 starting from Monday Jan 2, 2024
    capture_output(n_work_days("0201", 366), _).

:- end_tests(work_days).
