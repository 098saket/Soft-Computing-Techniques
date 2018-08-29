def readable_timedata(days):
    weeks = days // 7
    reaminder = days % 7
    return "{} Week(s) & {} Day(s).".format(days,reaminder)

print(readable_timedata(10))
