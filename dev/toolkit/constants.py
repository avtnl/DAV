from pathlib import Path
from datetime import date, datetime
import re

# Set-up paths using pathlib
INPUT_BASE = Path("C:/Users/avtnl/Documents/HU/orginele bestanden/")
OUTPUT_BASE = Path("C:/Users/avtnl/Documents/HU/Bestanden (output code)/")
PLOTS_BASE = OUTPUT_BASE / "Plots"
REPORTS_BASE = OUTPUT_BASE / "Reports"

# Fixed DATE_TIME used for all filenames in this run
DATE_TIME = datetime.now().strftime('%d%b%Y_%H%M')

# Maximum number of unique values to display in Explorer
MAX_UNIQUE_DISPLAY = 25

# Bank holidays list (provided by user)
list_of_bankholidays = [
    [(2016, 1, 1), (2016, 1, 18), (2016, 2, 15), (2016, 5, 30), None, (2016, 7, 4), (2016, 9, 5), (2016, 10, 10), (2016, 11, 11), (2016, 11, 24), (2016, 12, 26)],
    [(2017, 1, 2), (2017, 1, 16), (2017, 2, 20), (2017, 5, 29), None, (2017, 7, 4), (2017, 9, 4), (2017, 10, 9), (2017, 11, 10), (2017, 11, 23), (2017, 12, 25)],
    [(2018, 1, 1), (2018, 1, 15), (2018, 2, 19), (2018, 5, 28), None, (2018, 7, 4), (2018, 9, 3), (2018, 10, 8), (2018, 11, 12), (2018, 11, 22), (2018, 12, 25)],
    [(2019, 1, 1), (2019, 1, 21), (2019, 2, 18), (2019, 5, 27), None, (2019, 7, 4), (2019, 9, 2), (2019, 10, 14), (2019, 11, 11), (2019, 11, 28), (2019, 12, 25)],
    [(2020, 1, 1), (2020, 1, 20), (2020, 2, 17), (2020, 5, 25), None, (2020, 7, 3), (2020, 9, 7), (2020, 10, 12), (2020, 11, 11), (2020, 11, 26), (2020, 12, 25)],
    [(2021, 1, 1), (2021, 1, 18), (2021, 2, 15), (2021, 5, 31), (2021, 6, 18), (2021, 7, 5), (2021, 9, 6), (2021, 10, 11), (2021, 11, 11), (2021, 11, 25), (2021, 12, 24)],
    [(2022, 1, 1), (2022, 1, 17), (2022, 2, 21), (2022, 5, 30), (2022, 6, 20), (2022, 7, 4), (2022, 9, 5), (2022, 10, 10), (2022, 11, 11), (2022, 11, 24), (2022, 12, 26)],
    [(2023, 1, 2), (2023, 1, 16), (2023, 2, 20), (2023, 5, 29), (2023, 6, 19), (2023, 7, 4), (2023, 9, 4), (2023, 10, 9), (2023, 11, 10), (2023, 11, 23), (2023, 12, 25)],
    [(2024, 1, 1), (2024, 1, 15), (2024, 2, 19), (2024, 5, 27), (2024, 6, 19), (2024, 7, 4), (2024, 9, 2), (2024, 10, 14), (2024, 11, 11), (2024, 11, 28), (2024, 12, 25)],
    [(2025, 1, 1), (2025, 1, 20), (2025, 2, 17), (2025, 5, 26), (2025, 6, 19), (2025, 7, 4), (2025, 9, 1), (2025, 10, 13), (2025, 11, 11), (2025, 11, 27), (2025, 12, 25)]
]

# Build BANKHOLIDAYS_SET from list
BANKHOLIDAYS_SET = set(
    date(y, m, d)
    for year in list_of_bankholidays
    for entry in year
    if entry is not None
    for y, m, d in [entry]
)

# Define direction terms
# DIRECTION_TERMS = [
#     'north', 'east', 'south', 'west', 'n', 'e', 's', 'w',
#     'northeast', 'northwest', 'southeast', 'southwest', 'ne', 'nw', 'se', 'sw',
#     'northbound', 'eastbound', 'southbound', 'westbound', 'nb', 'sb', 'eb', 'wb',
#     'both ways', 'left', 'right', 'center'
# ]

DIRECTION_TERMS = {
    key: {"aliases": aliases, "format": key if key != "Rest" else None}
    for key, aliases in {
        "North": ["north", "n", "northbound", "nb"],
        "South": ["south", "s", "southbound", "sb"],
        "East": ["east", "e", "eastbound", "eb"],
        "West": ["west", "w", "westbound", "wb"],
        "NorthEast": ["northeast", "ne"],
        "SouthEast": ["southeast", "se"],
        "NorthWest": ["northwest", "nw"],
        "SouthWest": ["southwest", "sw"],
        "Rest": ["both ways", "left", "right", "center"],
    }.items()
}

# Precompile pattern for finding directions
DIRECTION_PATTERN = re.compile(
    r'(?:^|[^a-zA-Z])(' + '|'.join(re.escape(term) for term in DIRECTION_TERMS) + r')(?=[^a-zA-Z]|$)',
    re.IGNORECASE
)

ROAD_NAMES = {
    key: {'aliases': aliases, 'road_type': road_type, 'speed_type': speed_type, 'no_format': no_format, 'format': key if key != 'Rest' else None}
    for key, (aliases, road_type, speed_type, no_format) in {
        'Interstate': (['I-', 'IH-', 'Interstate', 'interstate'], 'Interstate', 'High Speed', 'I-'),
        'US Hwy': (['US Hwy', 'US-', 'U-', 'us highway'], 'US Highway', 'High Speed', 'US-'),
        'Fwy': (['Freeway', 'freeway', 'fway', 'fwy', 'fwyexit'], 'Freeway', 'High Speed', None),
        'Exprwy': (['Expressway', 'express', 'epwy', 'expy', 'expwy', 'exp', 'expyexit'], 'Expresswy', 'High Speed', None),
        'State Hwy': (['TX-', 'State Hwy', 'state highway'], 'State Highway', 'High Speed', 'TX-'),
        'Hwy': (['Hwy', 'highway', 'hway', 'hwy', 'hwyexit', 'entry ramp', 'exits', 'srvd', 'serv'], 'Highway', 'High Speed', None),
        'Tollwy': (['Tollway', 'tollway', 'toll', 'tlway', 'tlwy', 'tway', 'twy', 'tollwayexit', 'tollexpress', 'tollexit'], 'Tollway', 'High Speed', None),
        'Loop': (['Loop', 'loop', 'loops', 'loopway', 'state loop', 'loopexit'], 'Loop', 'High Speed', None),
        'FM': (['FM-', 'FR-', 'RM-', 'RR', 'fm', 'fr', 'rm', 'rr', 'farmrd', 'farmroad'], 'Farm-to-Market', 'High Speed', ['FM-', 'FR-', 'RM-', 'RR-']),
        'Parkwy': (['Parkway', 'parkway', 'parkwy', 'pway', 'pwy', 'pky', 'pkwyexit', 'pkyexit'], 'Parkway', 'High Speed', None),
        'Speedwy': (['Speedway', 'speedway', 'speedwy', 'sway', 'swy'], 'Speedway', 'High Speed', None),
        'Turnpke': (['Turnpike', 'turnpike', 'tpke', 'tpkeexit'], 'Turnpike', 'High Speed', None),
        'Blvd.': (['Blvd.', 'boulevard', 'blvd', 'blvdexit'], 'Boulevard', 'Low Speed', None),
        'Ave.': (['Ave.', 'avenue', 'ave', 'aveexit'], 'Avenue', 'Low Speed', None),
        'Way': (['Way', 'way', 'ways', 'wy', 'wayexit'], 'Way', 'Low Speed', None),
        'Ln.': (['Ln.', 'lane', 'la', 'ln', 'laneexit'], 'Lane', 'Low Speed', None),
        'Dr.': (['drive', 'dr', 'drexit'], 'Drive', 'Low Speed', None),
        'Str.': (['Str.', 'street', 'st', 'str', 'stexit'], 'Street', 'Low Speed', None),
        'Rd.': (['Rd.', 'road', 'rd', 'roadway', 'rdmain', 'rdexit'], 'Road', 'Low Speed', None),
        'Trl.': (['Trl.', 'trail', 'trl', 'trlexit'], 'Trail', 'Low Speed', None),
        'Pl.': (['Pl.', 'place', 'pl'], 'Place', 'Low Speed', None),
        'Plz.': (['Plz.', 'plaza', 'plz'], 'Plaza', 'Low Speed', None),
        'Ct.': (['Ct.', 'court', 'ct'], 'Court', 'Low Speed', None),
        'Ter.': (['Ter.', 'terrace', 'ter'], 'Terrace', 'Low Speed', None),
        'Rest': (['Main', 'Creek', 'Hill', 'Market', 'School', 'Church'], 'Rest', 'Low Speed', None),
    }.items()
}

INCIDENT_INFO = ['lane blocked', 'road closed due to accident', 'take alternate route', 'slow traffic', 'stationary traffic','one lane closed']

WEATHER_COLUMNS = [
    'Airport_Code', 'Weather_Timestamp', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
    'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition'
]

DAY_NIGHT_COLUMNS = ['Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']

DAY_TIMES = {
    'January':  {'1':  ('07:22 AM', '05:45 PM'), '15': ('07:18 AM', '05:54 PM')},
    'February': {'1':  ('07:14 AM', '06:03 PM'), '15': ('07:06 AM', '06:14 PM')},
    'March':    {'1':  ('07:00 AM', '06:21 PM'), '15': ('07:45 AM', '07:29 PM')},
    'April':    {'1':  ('07:10 AM', '07:40 PM'), '15': ('06:55 AM', '07:50 PM')},
    'May':      {'1':  ('06:40 AM', '08:00 PM'), '15': ('06:30 AM', '08:10 PM')},
    'June':     {'1':  ('06:25 AM', '08:25 PM'), '15': ('06:20 AM', '08:30 PM')},
    'July':     {'1':  ('06:25 AM', '08:30 PM'), '15': ('06:30 AM', '08:25 PM')},
    'August':   {'1':  ('06:40 AM', '08:15 PM'), '15': ('06:50 AM', '08:00 PM')},
    'September':{'1':  ('07:00 AM', '07:45 PM'), '15': ('07:10 AM', '07:30 PM')},
    'October':  {'1':  ('07:20 AM', '07:15 PM'), '15': ('07:30 AM', '07:00 PM')},
    'November': {'1':  ('07:40 AM', '06:45 PM'), '15': ('06:50 AM', '05:30 PM')},
    'December': {'1':  ('07:10 AM', '05:30 PM'), '15': ('07:15 AM', '05:35 PM')},
}

RAINY = [
    "Light Drizzle", "Light Rain", "Drizzle", "Rain", "Heavy Drizzle", "Heavy Rain",
    "Light Rain Shower", "Light Rain / Windy", "Light Rain with Thunder",
    "Light Thunderstorms and Rain", "Thunderstorms and Rain",
    "Heavy Thunderstorms and Rain", "Showers in the Vicinity"
]

SNOWY = [
    "Light Freezing Rain", "Light Freezing Drizzle", "Light Snow",
    "Light Snow / Windy", "Wintry Mix"
]

FOGGY = [
    "Patches of Fog", "Shallow Fog", "Fog", "Mist"
]

HAZY = [
    "Haze / Windy", "Widespread Dust"
]

WINDY = [
    "Cloudy / Windy", "Fair / Windy", "Partly Cloudy / Windy", "Mostly Cloudy / Windy",
    "Light Rain / Windy", "Rain / Windy", "Light Snow / Windy", "Haze / Windy"
]

STORMY = [
    "Light Thunderstorms and Rain", "T-Storm", "Thunderstorm", "Heavy T-Storm",
    "Heavy T-Storm / Windy", "Thunderstorms and Rain", "Heavy Thunderstorms and Rain"
]

SUNNY = [
    "Clear", "Fair", "Fair / Windy", "Partly Cloudy", "Partly Cloudy / Windy"
]

AIRPORT_COORDINATES = {
    'KMCJ': (29.71389, -95.39667), 'KDAL': (32.84594, -96.85088), 'KATT': (30.32078, -97.76043),
    'KADS': (32.96856, -96.83644), 'KRBD': (32.68131, -96.86878), 'KAUS': (30.19453, -97.66988),
    'KIAH': (29.98444, -95.34144), 'KHOU': (29.6458, -95.27723), 'KHQZ': (32.5205, -94.30778),
    'KSGR': (29.62225, -95.65653), 'KSAT': (29.53396, -98.46906), 'KELP': (31.80733, -106.37636),
    'KEDC': (29.50614, -95.47692), 'KDWH': (30.06178, -95.55279), 'KDFW': (32.89723, -97.03769),
    'KGTU': (30.67881, -97.67939), 'KBIF': (31.84953, -106.38005), 'KSPS': (33.9888, -98.4919),
    'KBAZ': (29.70575, -98.04322), 'KAMA': (35.21936, -101.70592), 'KJWY': (32.45829,-96.91253),
    'KTPL': (31.1519,  -97.40766), 'KINJ': (32.08364, -97.09725), 'KDZB': (30.52705, -98.35876),
    'KGGG': (32.384, -94.7115), 'KTRL': (32.70849, -96.26709), 'KLFK': (31.23403, -94.75),
    'KODO': (31.92142, -102.38713), 'KOSA': (33.09689, -94.96175), 'K6R3': (30.35644, -95.00803),
    'KBBD': (31.17928, -99.32392), 'KBPG': (32.21262, -101.52165), 'KCFD': (30.71569, -96.33136),
    'KCNW': (31.63781, -97.07414), 'KCPT': (32.35375, -97.43375), 'KGDJ': (32.44307, -97.82137),
    'KGLS': (29.26533, -94.86042), 'KLRD': (27.54419, -99.46158), 'KMFE': (26.17583, -98.23861),
    'KMWL': (32.78161, -98.06018), 'KOCH': (31.57776, -94.71011), 'KRFI': (32.14172, -94.85173),
    'KT20': (29.52912, -97.4643), 'KTXK': (33.45372, -93.99103), 'KACT': (31.61219, -97.23031),
    'KARM': (29.25428, -96.15439), 'KASL': (32.5205, -94.30778), 'KBGD': (35.70089, -101.39367),
    'KBKD': (32.71876, -98.8916), 'KBMT': (30.0702, -94.2151), 'KBRO': (25.90614, -97.426),
    'KCOT': (28.45572, -99.21722), 'KCRP': (27.77219, -97.50242), 'KCRS': (32.02806, -96.40058),
    'KCVB': (29.34239, -98.85122), 'KDLF': (29.35939, -100.77791), 'KDUA': (33.9397, -96.39506),
    'KDUX': (35.85743, -102.01331), 'KDYS': (32.41848, -98.5654), 'KFST': (30.91525, -102.91278),
    'KGDP': (31.83289, -104.80911), 'KGNC': (32.67533, -102.65267), 'KGRK': (31.06725, -97.82892),
    'KGYB': (30.16928, -96.98003), 'KGYI': (33.71419, -96.67436), 'KINK': (31.77981, -103.20169),
    'KJAS': (30.88569, -94.03494), 'KJDD': (32.74219, -95.49647), 'KLBR': (33.59264, -95.06415),
    'KORG': (30.06845, -93.80402), 'KPEQ': (31.38239, -103.51072), 'KPEZ': (28.95419, -98.51997),
    'KPPA': (35.613, -100.99625), 'KRKP': (28.08622, -97.04369), 'KSWW': (32.46736, -100.46656),
    'KT82': (30.24325, -98.90919), 'KUTS': (30.74689, -95.58717)
}