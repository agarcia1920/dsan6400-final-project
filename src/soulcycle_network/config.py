# This file contains the configuration for the SoulCycle network.
# It stores parameters that may be reused across the company-environment, rider-behavior, and network-formation modules. 

# Random seed to ensure reproducibility
RANDOM_SEED=6400


# TIME SETTINGS
# Unit of time for each interval of the simulation
TIME_STEP="week" #this model operates on a weekly time step

# Total number of weeks in the simulation
TOTAL_WEEKS=52 #one year

# Weekday configuration
# Standard weekday ordering
DAYS_OF_WEEK=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Numeric index for each day of the week
DAY_INDEX = {
    day: index
    for index, day in enumerate(DAYS_OF_WEEK)
}


# NODE BEHAVIOR SETTINGS
# Instructor availability configuration
# In our preliminary sample, we observed that instructors were off in a given week generally 8% of the time
PROB_OFF_WEEK=0.08

# Instructor deviation from normal classload 
MAX_WEEKLY_DEVIATION=3 #in our simulation we will constrain instructor classload to be within 3 classes of the normal classload

# Modeling assumption for instructor generation:
# one standard-deviation increase in baseline class load raises expected regular studio count by this many studios
CLASS_LOAD_STUDIO_EFFECT = 0.5


# Rider behavior configuration
# At the beginning of the simulation we will restrict riders to one class per day, but in the future we will allow riders to attend multiple classes per day
MAX_CLASSES_PER_DAY=1

# Simulated rider population size (scaled down from implied full-network demand)
TOTAL_SIMULATED_RIDERS=10000

# Target average occupancy when sizing the implied real-world rider population
TARGET_OCCUPANCY=0.70

# Approximate mean annual rides per active rider, used for population sizing
MEAN_ANNUAL_RIDES=12

# Persistent annual ride propensity parameters
# 27 annual rides is roughly the 90th percentile ("Top 10%"), so most riders attend less than once per week on average
# median is about 5 annual rides; distribution is right-skewed
RIDER_FREQUENCY_PARAMETERS={
    "log_mean": 1.6094379124341003, #log(5)
    "log_sd": 1.32,
    "minimum": 1,
    "maximum": 200,
}

# Analog network formation configuration
# Riders will need to attend 3 classes together to form a familiarity (weaker) connection
MIN_CLASSES_FOR_FAMILIARITY=3

# Riders will need to attend 6 classes together to form a social tie (stronger) connection
MIN_CLASSES_FOR_SOCIAL_TIE = 6

# Existing ties may weaken when riders stop encountering each other.
# A value of 0.95 means that 95% of the previous tie strength remains before new weekly encounters are added.
TIE_DECAY_RATE = 0.95

# Social ties require cumulative co-attendance and current tie strength after decay.
MIN_ACTIVE_TIE_STRENGTH_FOR_SOCIAL_TIE = 1.0

# Probability a rider may book at any studio in their home market rather than preferred/home-cluster studios only.
MARKET_WIDE_EXPLORATION_PROB = 0.05

# Rider coordination configuration
# Once riders form an analog social tie, they may coordinate future class attendance directly through ordinary offline communication, such as talking after class or texting privately.
# These limits prevent coordination from overwhelming independent rider preferences.
MAX_COORDINATION_PARTNERS_PER_WEEK = 2
MAX_COORDINATED_CLASSES_PER_WEEK = 2