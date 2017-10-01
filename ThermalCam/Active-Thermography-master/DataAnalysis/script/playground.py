from util import *
from multiprocessing import Pool

# material = 'cardboard' # 'stainlesssteel' # 'neoprene' # 'abs' # 'porcelain' # 'aluminum' #
# trial_num = 8 # [5,9]

# # play_trial(material, trial_num, 0.00000001, subtract_min=False, jump=10)

# # for material in materials:
# #     for trial_num in trial_nums:
# #         print material, trial_num
# #         create_video_from_trial(material, trial_num, '../1s/videos', use_min_pixel=False)

# # thermistor_plot_all_trial(material)

# # create_per_pixel_dataset(materials, trial_nums=10, base_name='dataset', subtract_min=False, num_pixels=500)

# # per_pixel_variance_plot(material, trial_num)
# create_video_from_trial(material, trial_num, path='', use_min_pixel=False)
# sys.exit(0)

# def work(args):
#     print args
#     m, trial = args
#     generate_DWT_Distance_matrix(m, trial, normalization=True, subtract_min=False, euclidean=False, downsampling=100, base_path=DTW_PATH)


# jobs = []
# for m in materials:
#     ts = check_existing_trials(m, DTW_PATH)
#     for trial in get_trials(m):
#         if trial in ts:
#             continue
#         print m, trial
#         jobs.append((m,trial))
#         # generate_DWT_Distance_matrix(m, trial, normalization=True, subtract_min=False, euclidean=False, downsampling=100, base_path=DTW_PATH)

# print jobs
# pool = Pool(processes=4)
# pool.map(work, jobs)
# for m in materials:
#     for trial in get_trials(m):
#         print m, trial
#         generate_ChebyshevDistance_matrix(m, trial, subtract_min=False, base_path=CHE_PATH)
#         # create_window_using_frame_after(m, trial, frame=900, path=WINDOW_PATH)


# material = 'cardboard'
# trial = 2
# timestamp, darr = load_trial_data(material, trial, False)
# temp = darr[:,41,323].copy()
# temp = resampling(timestamp, temp, pts=100)
# template = quick_normalized_model(np.linspace(0., TRIAL_INTERVAL, num=100))
# print DTWDist(normalize(temp), template)

# create_per_pixel_dataset(materials, base_name='hard', normalization=True, subtract_min=False, num_pixels=500)
# create_informative_dataset(materials, base_name='informative', normalization=True, subtract_min=False, binary=True)

# model = load_trained_model('../TrainedModels/hard.hdf5')
# # # model = load_pickle('../TrainedModels/svm_easy.pkl')
# for m in materials:
#     for t in get_trials(m):
#         print m, t
#         render_trial(m, t, model, FCN_PATH)

# m, t = ('6mat_1',0) # ('porcelain', 0) # ('neoprene', 0) # ('wood', 0) # ('neoprene', 0) # ('aluminum', 6) #
# display_class_activations(m, t, FCN_PATH)
# render_trial(m, t, model, FCN_PATH)
# classify_video(m,t,model)


#############
# 09/30/2017
material = 'castiron_30'
play_trial(material, 0, step=0.000001, subtract_min=True, normalization=False, jump=1)
# generate_ChebyshevDistance_matrix(material, 0, subtract_min=False, base_path=CHE_PATH)

# chemat = load_pickle(os.path.join(CHE_PATH, material, 'trial0.pkl'))
# thresh =  6 # 8 - 4mat_35
# chemat[chemat < thresh] = 0
# chemat[chemat >= thresh] = 200
# plt.matshow(chemat)
# plt.colorbar()
# plt.show()



