import numpy as np
from distfit import distfit
import scipy.stats as scipy

# input distribution as two arrays:
# nums: containing the numbers of people in each bucket_dwell range
# buckets: containing the strings of the ranges of each bucket_dwell range

def get_dwell_time(nums, buckets):
    filled_arr = np.empty([1, 1])
    for index, bucket in enumerate(buckets):
        lower, upper = map(int, bucket.split('-'))
        filled_arr = np.concatenate((filled_arr, np.random.uniform(low=lower, high=upper, size=(int(nums[index]), 1))))

    filled_arr = filled_arr[~np.isnan(filled_arr)]

    dist = distfit()
    dist.fit_transform(filled_arr)
    dist_name = dist.model['name']

    if not all(dist.model['arg']):
        loc_v, scale_v = dist.model['loc'], dist.model['scale']
        output_dwell_time = getattr(scipy, dist_name).rvs(loc=loc_v, scale=scale_v, size=1)
    elif len(dist.model['arg']) == 2:
        arg1, arg2, loc_v, scale_v = float((dist.model['arg'])[0]), float((dist.model['arg'])[1]), dist.model['loc'], \
                                     dist.model['scale']
        output_dwell_time = getattr(scipy, dist_name).rvs(arg1, arg2, loc=loc_v, scale=scale_v, size=1)
    else:
        arg, loc_v, scale_v = float((dist.model['arg'])[0]), dist.model['loc'], dist.model['scale']
        output_dwell_time = getattr(scipy, dist_name).rvs(arg, loc=loc_v, scale=scale_v, size=1)

    return output_dwell_time
