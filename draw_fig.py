import matplotlib.pyplot as plt
import os
import numpy as np

# cDLG_PSNR = []
# DLG_PSNR = []
# cDLG_result_x = []
# cDLG_result_y = []
# DLG_result_x = []
# DLG_result_y = []

# for pic in os.listdir('images'):
#     if 'cmp0.2' in pic:
#         if "cdlg" in pic:
#             cDLG_PSNR.append(float(pic.split('_')[1]))
# for pic in os.listdir('images'):
#     if 'cmp0.2' in pic:
#         if "dlg" in pic and "cdlg" not in pic:
#             DLG_PSNR.append(float(pic.split('_')[1]))
# print (np.mean(cDLG_PSNR))
# print (np.mean(DLG_PSNR))

# cDLG_result_x.append(0.2)
# cDLG_result_y.append(np.mean(cDLG_PSNR))
# DLG_result_x.append(0.2)
# DLG_result_y.append(np.mean(DLG_PSNR))

# cDLG_PSNR = []
# DLG_PSNR = []

# for pic in os.listdir('images'):
#     if 'cmp0.15' in pic:
#         if "cdlg" in pic:
#             cDLG_PSNR.append(float(pic.split('_')[1]))
# for pic in os.listdir('images'):
#     if 'cmp0.15' in pic:
#         if "dlg" in pic and "cdlg" not in pic:
#             DLG_PSNR.append(float(pic.split('_')[1]))
# print (np.mean(cDLG_PSNR))
# print (np.mean(DLG_PSNR))

# cDLG_result_x.append(0.15)
# cDLG_result_y.append(np.mean(cDLG_PSNR))
# DLG_result_x.append(0.15)
# DLG_result_y.append(np.mean(DLG_PSNR))

# cDLG_PSNR = []
# DLG_PSNR = []

# for pic in os.listdir('images'):
#     if 'cmp0.1' in pic:
#         if "cdlg" in pic:
#             cDLG_PSNR.append(float(pic.split('_')[1]))
# for pic in os.listdir('images'):
#     if 'cmp0.1' in pic:
#         if "dlg" in pic and "cdlg" not in pic:
#             DLG_PSNR.append(float(pic.split('_')[1]))
# print (np.mean(cDLG_PSNR))
# print (np.mean(DLG_PSNR))

# cDLG_result_x.append(0.1)
# cDLG_result_y.append(np.mean(cDLG_PSNR))
# DLG_result_x.append(0.1)
# DLG_result_y.append(np.mean(DLG_PSNR))

# cDLG_PSNR = []
# DLG_PSNR = []

# for pic in os.listdir('images'):
#     if 'cmp0.05' in pic:
#         if "cdlg" in pic:
#             cDLG_PSNR.append(float(pic.split('_')[1]))
# for pic in os.listdir('images'):
#     if 'cmp0.05' in pic:
#         if "dlg" in pic and "cdlg" not in pic:
#             DLG_PSNR.append(float(pic.split('_')[1]))
# print (np.mean(cDLG_PSNR))
# print (np.mean(DLG_PSNR))

# cDLG_result_x.append(0.05)
# cDLG_result_y.append(np.mean(cDLG_PSNR))
# DLG_result_x.append(0.05)
# DLG_result_y.append(np.mean(DLG_PSNR))

# cdlg_line = plt.plot(cDLG_result_x, cDLG_result_y, label='our method')
# dlg_line = plt.plot(DLG_result_x, DLG_result_y, label="Geiping's")
# plt.legend()

# plt.xlabel('sparsity')
# plt.ylabel('PSNR')

# ax = plt.gca()
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# plt.show()


sim = ["0.7500323492984824", "0.7763739209198292", "0.7891934857755514", "0.7966060964563654", "0.8041989389430098", "0.8049522154648132", "0.8110200195944323", "0.8095689224910808", "0.8124202820858828", "0.8138806218459435", "0.8199345619904985", "0.8202118416917759", "0.8218385492726029", "0.8211361073627004", "0.8218801412277945", "0.8229661533911308", "0.8260439580753092", "0.8264321496570974", "0.827587481745753", "0.8264275283287428", "0.8290524428341682", "0.8281420411483077", "0.8293158585503817", "0.8320747915780912", "0.8355731371425402", "0.8360953472466126", "0.8331376970996544", "0.8369687783056362", "0.8354160119784831", "0.8356101077693773", "0.8363680056195353", "0.8387017764386195", "0.8425421003013106", "0.8406704623176886", "0.8422555779433241", "0.8423387618537073", "0.8409985766308667", "0.8396214207811894", "0.8410725178845407", "0.8413913895410097", "0.8383644194687321", "0.8398155165720835", "0.8414838161081021", "0.8410632752278315", "0.8418951143316635", "0.8424034604506719", "0.8455459637318151", "0.844862007135331", "0.8457123315525815", "0.8441965358522654", "0.8440948666284637", "0.845559827716879", "0.847283583193153", "0.8508743553246946", "0.8504214651459415", "0.8508281420411483", "0.8516969517718173", "0.8542294397101503", "0.8551768120228479", "0.8544928554263638", "0.8507264728173466", "0.8509991311902694", "0.8521128713237333", "0.8510176165036878", "0.8511285283841987", "0.8499916816089617", "0.8507264728173466", "0.8523993936817199", "0.8515259626226963", "0.8533467659944174", "0.8553847717988059", "0.856789655618611", "0.8536148030389855", "0.8530879716065586", "0.8501765347431466", "0.84989001238516", "0.8515074773092778", "0.8490396879679095", "0.8490443092962641", "0.8508512486829214", "0.8487901362367599", "0.8507264728173466", "0.8495526554152726", "0.8518355916224559", "0.8527783426067989", "0.8499362256687062", "0.8537811708597519", "0.8541139065012847", "0.8535269978002477", "0.8543357302623066"]
iteration = 0
for s in sim:
    sim[iteration] = round(float(s), 4)
    iteration+=1

sim_line = plt.plot(range(0, len(sim)), sim)
plt.legend()

# plt.ylim(0, 1)
plt.xlabel('iter')
plt.ylabel('similarity')

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()