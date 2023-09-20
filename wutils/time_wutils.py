import pandas as pd


def month_index(time):
    return pd.DatetimeIndex(pd.Series(time)).month


def season_index(month_idx, hemisphere='S'):

    if hemisphere=='N':
        season_dict = {'1': 'Winter',
                    '2': 'Winter',
                    '3': 'Spring', 
                    '4': 'Spring',
                    '5': 'Spring',
                    '6': 'Summer',
                    '7': 'Summer',
                    '8': 'Summer',
                    '9': 'Autumn',
                    '10': 'Autumn',
                    '11': 'Autumn',
                    '12': 'Winter'}
    else:
        season_dict = {'1': 'Summer',
                    '2': 'Summer',
                    '3': 'Autumn', 
                    '4': 'Autumn',
                    '5': 'Autumn',
                    '6': 'Winter',
                    '7': 'Winter',
                    '8': 'Winter',
                    '9': 'Spring',
                    '10': 'Spring',
                    '11': 'Spring',
                    '12': 'Summer'}

    return month_idx.to_series().apply(lambda x: season_dict[str(x)]).values

