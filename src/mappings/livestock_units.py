'''
This module maps agricalc cattle types to Livestock Units (LU) for the purpose of diet calculations
'''

AC_beef_types=['Bull', #Types used in aregaalc
            'Entire 0-12 mnth',
            'Entire 12-24 mnth',
            'Heifer  0-12 mnth',
            'Heifer 12-24 mnth',
            'Heifer 24-36 mnth',
            'Steer  0-12 mnth',
            'Steer  24-36 mnth',
            'Steer 12-24 mnth',
            'Suckler cow']

AC_LU_mapping={'Bull':0.65, #From https://www.fas.scot/downloads/fmh-october-2024-livestock/
            'Entire 0-12 mnth':0.34, 
            'Entire 12-24 mnth':0.65,
            'Heifer  0-12 mnth':0.34,
            'Heifer 12-24 mnth':0.65,
            'Heifer 24-36 mnth':0.8,
            'Steer  0-12 mnth':0.34,
            'Steer  24-36 mnth':0.8,
            'Steer 12-24 mnth':0.65,
            'Suckler cow':0.75} 