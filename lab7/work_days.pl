% work_days.pl
% Solution for calculating N working days from a given date in 2024

% Days in each month of 2024 (non-leap year)
days_in_month(1, 31).
days_in_month(2, 28).  % switch to non-leap year 2024
days_in_month(3, 31).
days_in_month(4, 30).
days_in_month(5, 31).
days_in_month(6, 30).
days_in_month(7, 31).
days_in_month(8, 31).
days_in_month(9, 30).
days_in_month(10, 31).
days_in_month(11, 30).
days_in_month(12, 31).

% Day names for 2024: use 1=Monday..5=Friday, 6=Saturday, 0=Sunday
day_name(1, "Monday").
day_name(2, "Tuesday").
day_name(3, "Wednesday").
day_name(4, "Thursday").
day_name(5, "Friday").
day_name(6, "Saturday").
day_name(0, "Sunday").

% Convert date string to numbers
parse_date(DateStr, Day, Month) :-
    sub_string(DateStr, 0, 2, _, DayStr),
    sub_string(DateStr, 2, 2, _, MonthStr),
    number_string(Day, DayStr),
    number_string(Month, MonthStr).

% Convert numbers back to string format with leading zeros
format_date(Day, Month, DateStr) :-
    format(string(DayStr), "~|~`0t~d~2+", [Day]),
    format(string(MonthStr), "~|~`0t~d~2+", [Month]),
    string_concat(DayStr, MonthStr, DateStr).

% Calculate day of week (0-6, where 1=Monday) based on Jan 1, 2024 being Sunday (0)
day_of_week(Day, Month, WeekDay) :-
    days_before_month(Month, DaysBefore),
    Total is DaysBefore + Day - 1,
    WeekDay is Total mod 7.

% Calculate total days before a given month
days_before_month(1, 0).
days_before_month(Month, Days) :-
    Month > 1,
    PrevMonth is Month - 1,
    days_before_month(PrevMonth, PrevDays),
    days_in_month(PrevMonth, MonthDays),
    Days is PrevDays + MonthDays.

% Add working days to a date
add_work_days(Day, Month, 0, Day, Month, WeekDay) :-
    day_of_week(Day, Month, WeekDay),
    WeekDay >= 1, WeekDay =< 5.
add_work_days(Day, Month, N, ResultDay, ResultMonth, ResultWeekDay) :-
    N > 0,
    next_day(Day, Month, NextDay, NextMonth),
    day_of_week(NextDay, NextMonth, NextWeekDay),
    ( NextWeekDay >= 1, NextWeekDay =< 5 ->
        N1 is N - 1
    ;   N1 = N ),
    add_work_days(NextDay, NextMonth, N1, ResultDay, ResultMonth, ResultWeekDay).

% Calculate next day
next_day(Day, Month, NextDay, Month) :-
    days_in_month(Month, MaxDays),
    Day < MaxDays,
    NextDay is Day + 1.
next_day(Day, Month, 1, NextMonth) :-
    days_in_month(Month, MaxDays),
    Day >= MaxDays,
    Month < 12,
    NextMonth is Month + 1.
next_day(Day, 12, 1, 1) :-  % Handle year wrap
    days_in_month(12, MaxDays),
    Day >= MaxDays.

% Compute the previous calendar day
prev_day(Day, Month, PrevDay, Month) :-
    Day > 1,
    PrevDay is Day - 1.
prev_day(Day, Month, PrevDay, PrevMonth) :-
    Day =:= 1,
    Month > 1,
    PrevMonth is Month - 1,
    days_in_month(PrevMonth, PrevDay).
prev_day(Day, 1, PrevDay, 12) :-  % handle year wrap if needed
    Day =:= 1,
    days_in_month(12, PrevDay).

% Compute the previous working day (skip weekends)
previous_workday(Day, Month, PWDay, PWMonth) :-
    prev_day(Day, Month, PD, PM),
    day_of_week(PD, PM, WD),
    WD < 5,
    PWDay = PD,
    PWMonth = PM.
previous_workday(Day, Month, PWDay, PWMonth) :-
    prev_day(Day, Month, PD, PM),
    day_of_week(PD, PM, WD),
    WD >= 5,
    previous_workday(PD, PM, PWDay, PWMonth).

% Main predicate
n_work_days(DateStr, N) :-
    N =< 366,
    parse_date(DateStr, Day, Month),
    % start date must be a working day (Mon-Fri)
    day_of_week(Day, Month, StartWD),
    StartWD >= 1, StartWD =< 5,
    add_work_days(Day, Month, N, RD, RM, WD),
    day_name(WD, DayName),
    format_date(RD, RM, DateStrOut),
    format("~w, ~w~n", [DayName, DateStrOut]).
