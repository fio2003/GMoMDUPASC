#!/usr/bin/python
def prepare_data():
    import numpy as np
    in_file = "sincos.dat"
    num_elements = 1000
    verbose = 2
    file = open(in_file, 'rb')
    initial_1d_array = np.frombuffer(file.read(), dtype=np.float64, count=-1)
    file.close()
    good_2d_matr = np.reshape(initial_1d_array, (num_elements, -1))
    X = good_2d_matr
    in_file = "clusters.dat"
    with open(in_file) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
        clusters = [x.strip() for x in content]

    full_clusters = list()
    for i in range(10):
        full_clusters.append(list())


    for i in range(num_elements):
        full_clusters[int(clusters[i])-1].append(X[i])

    for i in range(10):
        with open(str(i+1)+"_elements.dat", 'w') as f:
            for cur_points in full_clusters[i]:
                for point in cur_points:
                    f.write(str(point) + ' ')
                f.write('\n')


def get_pca_data(clean_numbers_np, threshold = 0):

    clean_numbers_np_T = clean_numbers_np.T
    clean_numbers_np_T_new = list()
    pmeans_y = list()
    for line in clean_numbers_np_T:
        cur_mean = np.mean(line)
        pmeans_y.append(cur_mean)
        clean_numbers_np_T_new.append(line - cur_mean)

    clean_numbers_np_T_new_np_T = np.array(clean_numbers_np_T_new)
    del line, cur_mean, clean_numbers_np, clean_numbers_np_T

    cov = np.cov(clean_numbers_np_T_new_np_T)
    from scipy import linalg
    ut, s, vN = linalg.svd(cov, full_matrices=True, lapack_driver='gesvd')

    eigen_inv = np.linalg.inv(vN.T)
    for i in range(len(s)):
        if s[i] < threshold:
            s[i] = threshold
    current_sum = 0
    saved_i = -1
    total_sum = sum(s)
    # for i in range(76):
    #     current_sum += s[i]
    #     if i > 0 and (current_sum / total_sum > 0.99):
    #         saved_i = i
    #         print('For cluster ' + cur_index + ' found ' + str(saved_i) + ' significant eigenvalues, resulting in '
    #               + str(100*current_sum/total_sum) + '% of variance')
    #         break
    saved_i = 76
    if saved_i <= 0:
        exit('very unexpected error')
    top_eigenvalues = np.array(s[0:saved_i].T)
    top_eigenvectors = np.array(vN[0:saved_i].T)
    full_eigenvectors = np.concatenate( (top_eigenvectors, np.zeros((top_eigenvectors.shape[0], top_eigenvectors.shape[0] - saved_i))), axis=1)

    two_pi_eigen = 2 * math.pi * top_eigenvalues
    det_root = math.sqrt(np.prod(two_pi_eigen))

    return (saved_i, pmeans_y, top_eigenvalues, top_eigenvectors, eigen_inv, full_eigenvectors, s, det_root)


def get_min_max():
    import numpy as np
    in_file = "sincos.dat"
    file = open(in_file, 'rb')
    initial_1d_array = np.frombuffer(file.read(), dtype=np.float64, count=-1)
    file.close()
    good_2d_matr = np.reshape(initial_1d_array, (1000, -1))
    loc_min = list()
    loc_max = list()
    for i in range(76):
        loc_min.append(min(good_2d_matr[i]) )
        loc_max.append(max(good_2d_matr[i]) )
    return np.array(loc_min), np.array(loc_max)

def check_points2(points, eigen_val, bot_part):
    import math
    import numpy as np
    # two_pi_eigen= 2*math.pi*eigen_val
    # det_root = math.sqrt(np.prod(two_pi_eigen))

    # np_points = np.array(points)

    top_part = math.exp(-0.5*np.sum(points*points/eigen_val)) # 1x76 * 76x76 * 76x1

    # if top_part == 0:
    #     return 0
    prob = top_part / bot_part
    return prob

def get_data(cur_index):
    import numpy as np
    cur_index = str(cur_index)
    # in_file = 'processed_' + cur_index + "_elements.dat"
    in_file = cur_index + "_elements.dat"
    with open(in_file, 'r') as f:
        content = f.readlines()
    del in_file, f

    clean_numbers = list()

    # for line in content:
    #     points_set = list(map(float, line[:-2].strip().split(',')))
    #     clean_numbers.append(points_set)
    for line in content:
        points_set = list(map(float, line.strip().split(' ')))
        clean_numbers.append(points_set)
    del points_set

    clean_numbers_np = np.array(clean_numbers)
    return clean_numbers_np

def get_threshhold(eigenvalues, method):
    if method == 'arith':
        import numpy as np
        avg = np.average(eigenvalues)
    elif method == 'geom':
        import scipy
        from scipy import stats
        from scipy.stats import mstats
        from scipy.stats.mstats import gmean
        avg = gmean(eigenvalues)
    elif method == 'harm':
        import scipy
        from scipy import stats
        from scipy.stats import mstats
        from scipy.stats.mstats import hmean
        avg = hmean(eigenvalues)
    else:
        print('Unknown method for average')
        exit(-1)

    return avg

if __name__ == "__main__":
    import numpy as np
    import math
    import argparse
    with open("README.txt") as f:
        content = f.readlines()
    
    readme = str()
    for line in content:
        readme += line

    parser = argparse.ArgumentParser(description=readme)
    parser.add_argument('-g', '--generate', help='generate points with Metropolis step', type=int)
    args = parser.parse_args()
    if not args.generate:
        prepare_data()
        exit(0)
    tot_points_to_generate = int(args.generate)
    # read cluster points
    dat = list()
    for i in range(10):
        dat.append(get_data(i+1))

    # combine all points to get threshold(thermal noise)
    full_set = dat[0]
    for i in range(9):
        full_set = np.concatenate((full_set, dat[i+1]))

    all_points_pca_data = get_pca_data(full_set)
    all_points_pca_data_eigval = all_points_pca_data[2]
    threshold = get_threshhold(all_points_pca_data_eigval, 'geom') #arith, geom, harm

    # get data for each cluster
    pca_data = list()
    for i in range(10):
        pca_data.append(get_pca_data(dat[i], threshold))


    # Compute algorithm energy
    alg_clus_energy = np.zeros(10)
    for point in full_set:
        i = 0
        for cluster in pca_data:
            centering_arr = cluster[1]
            centered = np.array(point - centering_arr)  # move point to the particular cluster's center
            reduced_eigenv = cluster[3]
            rotated = centered.dot(reduced_eigenv)

            # cluster[2] - variance, cluster[7] - precomputed bot_part
            cluster_energy = -check_points2(rotated, cluster[2], cluster[7])
            if cluster_energy < -1.0:
                #print('Point index ',np.where(full_set == point)[0][0])
                #print('Point values ', full_set[np.where(full_set == point)[0][0]])
                #print('Cluster index ', i)
                check_points2(rotated, cluster[2], cluster[7])
                # print('Cluster values ', pca_data[i])
            alg_clus_energy[i] += cluster_energy
            i += 1
    print('Each cluster energy: ')
    for i in range(len(alg_clus_energy)):
        print(str(i) + ': %.4g' % alg_clus_energy[i])
    print('Total alg energy: %.4g, Energy per point: %.4g' % (np.sum(alg_clus_energy), np.sum(alg_clus_energy) / len(full_set)))

    # exit(0)
    # Simulation part

    loc_min, loc_max = get_min_max()
    step_size = (loc_max - loc_min)/100.0

    t_var = 1e-13 # temperature variable for metropolis criterion

    # get initial point
    cur_point = np.random.normal(scale=np.sqrt(pca_data[0][6])) #generate point with first cluster, first component variance
    reduced_dim_set = pca_data[0][4].T.dot(cur_point) #multiply eigenvectors and point
    point = cur_point.T.dot(pca_data[0][3]).T[0] #multiply by inverce eigenvector
    point += pca_data[0][1] #translate points by adding normalization coefficients

    saved_points = list() #list with points that somehow passed either better condition critiria or metropolis  criteria
    algorithm_energy = 0
    cur_energy = 0 #initialization of the variable
    new_energy = 0 #initialization of the variable

    # Prestep
    for cluster in pca_data:
        centering_arr = cluster[1]
        centered = np.array(point - centering_arr) #move point to the particular cluster's center
        reduced_eigenv = cluster[3]
        rotated = centered.dot(reduced_eigenv)

        cluster_energy = -check_points2(rotated, cluster[2], cluster[7]) # cluster[2] - variance, cluster[7] - precomputed bot_part
        new_energy += cluster_energy
    algorithm_energy += new_energy
    if new_energy < cur_energy:
        saved_points.append(point)
    else:
        metr_criterion = math.exp( (cur_energy - new_energy)/t_var)
        myrand = np.random.uniform()
        if myrand < metr_criterion:
            saved_points.append(point)

    total_steps = tot_points_to_generate 
    cluster_visit = np.zeros(len(pca_data))
    visit_map = [0, ]
    metr_accepted = 0

    # Main loop
    for step in range(total_steps):
        for i in range(len(point)):
            point[i] += np.random.normal(0, step_size[i])
        cur_energy = new_energy
        new_energy = 0

        j = 0
        en_arr = list()
        for cluster in pca_data:
            centering_arr = cluster[1]
            centered = np.array(point - centering_arr)
            reduced_eigenv = cluster[3]
            rotated = centered.dot(reduced_eigenv)

            cluster_energy = -check_points2(rotated, cluster[2], cluster[7])
            en_arr.append(cluster_energy)
            new_energy += cluster_energy

        j = en_arr.index(max(en_arr))
        cluster_visit[j] += 1
        if visit_map[-1] != j:
            visit_map.append(j)

        if new_energy < cur_energy:
            saved_points.append(point)
            algorithm_energy += new_energy
        else:
            metr_criterion = math.exp((cur_energy - new_energy) / t_var)
            myrand = np.random.uniform()
            if myrand < metr_criterion:
                saved_points.append(point)
                algorithm_energy += new_energy
                metr_accepted += 1
            else:
                point = saved_points[-1]
    print('Wrong acceptance : ' + str(100 * metr_accepted / (total_steps+1)) + '%  - steps accepted with metropolis criterion')
    print('Acceptance rate : ' + str(100*len(saved_points)/(total_steps+1)) +'%')
    print('Algorithm energy: ' + str(algorithm_energy) + ' Corrected: ' + str(algorithm_energy/total_steps))

    print('Cluster visits: \n')
    for elem in cluster_visit:
        print(str(elem))

    print('Visit map: ' + str(visit_map))

