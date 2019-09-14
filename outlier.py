import numpy as np

def MahalanobisDist(data):
    '''Computing covariance matrix for the data'''
    covariance_matrix = np.cov(data, rowvar=False)

    '''Computing pseudoinverse of the covariance matrix'''
    inv_covariance_matrix = np.linalg.pinv(covariance_matrix)        
    vars_mean = []
    
    '''Appending mean of the columns respectively in vars_mean'''
    for i in range(data.shape[0]):
        vars_mean.append(list(data.mean(axis=0)))

    diff = data - vars_mean
    md = []

    '''Computing Mahalanobis distance for each feature'''
    for i in range(len(diff)):
        md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
    
    return md


def MD_detectOutliers(data, extreme=False):

    MD = MahalanobisDist(data)

    '''one popular way to specify the threshold'''
    m = np.mean(MD)
    t = 3. * m if extreme else 2. * m
    outliers = []
    for i in range(len(MD)):
       if MD[i] > t:
           outliers.append(i)  # index of the outlier
    return np.array(outliers)

    '''
    or according to the 68–95–99.7 rule

    std = np.std(MD)
    k = 3. * std if extreme else 2. * std
    m = np.mean(MD)
    up_t = m + k
    low_t = m - k
    outliers = []
    for i in range(len(MD)):
        if (MD[i] >= up_t) or (MD[i] <= low_t):
            outliers.append(i)  # index of the outlier
    return np.array(outliers)
    '''

