"""
locations.py

Peninsular Malaysia state and city coordinate lookup used to auto fetch
NASA POWER historical weather and the Open Meteo forecast for whichever
location the user selects in the sidebar.
"""

MALAYSIA_LOCATIONS = {
    "Selangor": {
        "Shah Alam":     (3.0733, 101.5185),
        "Petaling Jaya": (3.1073, 101.6067),
        "Klang":         (3.0449, 101.4455),
        "Kajang":        (2.9931, 101.7874),
        "Rawang":        (3.3172, 101.5764),
    },
    "Kuala Lumpur": {
        "KL City Centre": (3.1390, 101.6869),
        "Cheras":          (3.1010, 101.7414),
        "Kepong":          (3.2107, 101.6350),
    },
    "Putrajaya": {
        "Putrajaya City": (2.9264, 101.6964),
    },
    "Negeri Sembilan": {
        "Seremban":     (2.7297, 101.9381),
        "Port Dickson": (2.5222, 101.7959),
        "Nilai":        (2.8064, 101.7975),
    },
    "Melaka": {
        "Melaka City": (2.1896, 102.2501),
        "Alor Gajah":  (2.3785, 102.2088),
    },
    "Johor": {
        "Johor Bahru": (1.4927, 103.7414),
        "Batu Pahat":  (1.8548, 102.9325),
        "Muar":        (2.0442, 102.5689),
        "Kluang":      (2.0312, 103.3183),
    },
    "Pahang": {
        "Kuantan":  (3.8168, 103.3317),
        "Temerloh": (3.4508, 102.4193),
        "Bentong":  (3.5225, 101.9089),
    },
    "Terengganu": {
        "Kuala Terengganu": (5.3117, 103.1324),
        "Dungun":           (4.7570, 103.4197),
    },
    "Kelantan": {
        "Kota Bharu": (6.1254, 102.2381),
        "Pasir Mas":  (6.0453, 102.1394),
    },
    "Perak": {
        "Ipoh":        (4.5975, 101.0901),
        "Taiping":     (4.8500, 100.7333),
        "Teluk Intan": (4.0227, 101.0208),
    },
    "Penang": {
        "George Town": (5.4141, 100.3288),
        "Bayan Lepas": (5.2945, 100.2570),
        "Butterworth": (5.3991, 100.3638),
    },
    "Kedah": {
        "Alor Setar":    (6.1184, 100.3685),
        "Sungai Petani": (5.6467, 100.4876),
        "Langkawi":      (6.3500, 99.8000),
    },
    "Perlis": {
        "Kangar": (6.4414, 100.1986),
    },
}
