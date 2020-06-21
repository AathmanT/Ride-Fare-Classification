import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import tensorflow as tf

df = pd.read_csv("train.csv")
df2 = pd.read_csv("test.csv")

#####################################
####### Handling missing values#####
#####################################

df = df.replace('nan',np.NaN)
df.dropna(subset=['additional_fare','duration','meter_waiting','meter_waiting_fare','meter_waiting_till_pickup'], thresh=5,inplace=True)

df.fillna(df.mean().iloc[0],inplace=True)

y_train = df.iloc[:,-1].values
y_tr = df.iloc[:,-1]

df = df.drop("label", axis=1)

####################################
#####    New features  #############
####################################

a = np.sin((np.radians(df['drop_lat'])-np.radians(df['pick_lat'])) / 2) ** 2 + np.cos(df['drop_lat']) * np.cos(df['pick_lat']) * np.sin((np.radians(df['drop_lon'])-np.radians(df['pick_lon']))/ 2) ** 2
c = 2 * np.arcsin(np.sqrt(a))
kilometers = 6371.0 * c
df['trip_distance'] = kilometers


a = np.sin((np.radians(df2['drop_lat'])-np.radians(df2['pick_lat'])) / 2) ** 2 + np.cos(df2['drop_lat']) * np.cos(df2['pick_lat']) * np.sin((np.radians(df2['drop_lon'])-np.radians(df2['pick_lon']))/ 2) ** 2
c = 2 * np.arcsin(np.sqrt(a))
kilometers = 6371.0 * c
df2['trip_distance'] = kilometers


df['effective_duration'] = df['duration']-df['meter_waiting']
df2['effective_duration'] = df2['duration']-df2['meter_waiting']



#####################################################
#########  More New features no improvemnt  #########
#####################################################

# df.insert(7, 'total_duration', df['duration']+df['meter_waiting_till_pickup'])
# df2.insert(7, 'total_duration', df2['duration']+df2['meter_waiting_till_pickup'])

# df.insert(8, 'effective_fare', df['fare']-df['meter_waiting_fare'])
# df2.insert(8, 'effective_fare', df2['fare']-df2['meter_waiting_fare'])



####################################
#####    New features  #############
####################################


df['effective_fare'] =  df['fare']-df['meter_waiting_fare']-df['additional_fare']
df2['effective_fare'] = df2['fare']-df2['meter_waiting_fare']-df['additional_fare']


df['pickup_time'] = pd.to_datetime(df['pickup_time'])
df['drop_time'] = pd.to_datetime(df['drop_time'])

df['day'] = df['pickup_time'].dt.dayofweek
count1 = df['day'].value_counts()

cut_labels_4 = ['weekday', 'weekend']
cut_bins = [-np.inf, 4, 6]
df['day_bin'] = pd.cut(df['day'], bins=cut_bins, labels=cut_labels_4)
df = df.drop("day", axis=1)

tem1 = pd.get_dummies(df.day_bin, prefix='day')
tem1 = tem1.drop("day_weekend", axis=1)
df = pd.concat([df, tem1], axis=1, sort=False)
df = df.drop("day_bin", axis=1)



#####################################################
#########  More New features no improvemnt  #########
#####################################################

# df['hour'] = df['pickup_time'].dt.hour
# count2 = df['hour'].value_counts()
#
# cut_labels_4 = ['cat1', 'cat2','cat3', 'cat4','cat5', 'cat6','cat7', 'cat8']
# cut_bins = [-np.inf,3,5,8,11,15,18,21,24]
# df['hour_bin'] = pd.cut(df['hour'], bins=cut_bins, labels=cut_labels_4)
# df = df.drop("hour", axis=1)
#
# # tem2 = pd.get_dummies(df.hour_bin, prefix='hour')
# # tem2 = tem2.drop("hour_cat8", axis=1)
# # df = pd.concat([df, tem2], axis=1, sort=False)
# # df = df.drop("hour_bin", axis=1)
#
# # day_x_hour = tf.feature_column.crossed_column([df.hour_bin, df.day_bin], hash_bucket_size=1000)
# df['day_x_hour'] = df['day_bin'].astype(str)+df['hour_bin'].astype(str)
# df = df.drop("day_bin", axis=1)
# df = df.drop("hour_bin", axis=1)
#
# tem2 = pd.get_dummies(df.day_x_hour, prefix='d_x_hour')
# tem2 = tem2.drop("d_x_hour_weekendcat8", axis=1)
# df = pd.concat([df, tem2], axis=1, sort=False)
# df = df.drop("day_x_hour", axis=1)



####################################
#####    New features  #############
####################################


df2['pickup_time'] = pd.to_datetime(df2['pickup_time'])
df2['drop_time'] = pd.to_datetime(df2['drop_time'])

df2['day'] = df2['pickup_time'].dt.dayofweek
count1 = df2['day'].value_counts()

cut_labels_4 = ['weekday', 'weekend']
cut_bins = [-np.inf, 4, 6]
df2['day_bin'] = pd.cut(df2['day'], bins=cut_bins, labels=cut_labels_4)
df2 = df2.drop("day", axis=1)

tem1 = pd.get_dummies(df2.day_bin, prefix='day')
tem1 = tem1.drop("day_weekend", axis=1)
df2 = pd.concat([df2, tem1], axis=1, sort=False)
df2 = df2.drop("day_bin", axis=1)



#####################################################
#########  More New features no improvemnt  #########
#####################################################

#
# df2['hour'] = df2['pickup_time'].dt.hour
# count2 = df2['hour'].value_counts()
#
# cut_labels_4 = ['cat1', 'cat2','cat3', 'cat4','cat5', 'cat6','cat7', 'cat8']
# cut_bins = [-np.inf,3,5,8,11,15,18,21,24]
# df2['hour_bin'] = pd.cut(df2['hour'], bins=cut_bins, labels=cut_labels_4)
# df2 = df2.drop("hour", axis=1)
#
# # tem2 = pd.get_dummies(df2.hour_bin, prefix='hour')
# # tem2 = tem2.drop("hour_cat8", axis=1)
# # df2 = pd.concat([df2, tem2], axis=1, sort=False)
# # df2 = df2.drop("hour_bin", axis=1)
#
# df2['day_x_hour'] = df2['day_bin'].astype(str)+df2['hour_bin'].astype(str)
# df2 = df2.drop("day_bin", axis=1)
# df2 = df2.drop("hour_bin", axis=1)
#
# tem2 = pd.get_dummies(df2.day_x_hour, prefix='d_x_hour')
# tem2 = tem2.drop("d_x_hour_weekendcat8", axis=1)
# df2 = pd.concat([df2, tem2], axis=1, sort=False)
# df2 = df2.drop("day_x_hour", axis=1)





df = df.drop(columns=['pickup_time', 'drop_time'])
df2 = df2.drop(columns=['pickup_time', 'drop_time'])


X_train = df.iloc[:,1:].values


print('nan columns',df.columns[df.isnull().any()])
print("")
print('nan rows',df[df.isna().any(axis=1)])

y_train[y_train=='correct'] = 1
y_train[y_train=='incorrect'] = 0

##################################################
####### Feature Importance no improvment #########
##################################################

# # use feature importance for feature selection, with fix for xgboost 1.0.2
# from numpy import loadtxt
# from numpy import sort
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import SelectFromModel
#
# # define custom class to fix bug in xgboost 1.0.2
# class MyXGBClassifier(XGBClassifier):
# 	@property
# 	def coef_(self):
# 		return None
#
#
# # split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=7)
# # fit model on all training data
# model = MyXGBClassifier(learning_rate =0.01,
#  n_estimators=5000,
#  max_depth=4,
#  min_child_weight=6,
#  gamma=0,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  reg_alpha=0.005,
#  objective= 'binary:logistic',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27)
# model.fit(X_train, y_train)
# # make predictions for test data and evaluate
# predictions = model.predict(X_test)
#
# from sklearn.metrics import fbeta_score
# fbeta = fbeta_score(y_test.astype(int), predictions.astype(int), average='macro', beta=0.5)
# # accuracy = accuracy_score(y_test, predictions)
# print("fbeta: %.2f%%" % (fbeta * 100.0))
# # Fit model using each importance as a threshold
# thresholds = sort(model.feature_importances_)
# print(thresholds)
#
#
# #
# # X_train = df.iloc[:,1:].values
# # X_tr = df.iloc[:,1:]
# #
# # y_train = df.iloc[:,-1].values
# # y_tr = df.iloc[:,-1]
# #
# # y_train[y_train=='correct'] = 1
# # y_train[y_train=='incorrect'] = 0
# #
# # X_test = df2.iloc[:,1:].values
# # X_te = df2.iloc[:,1:]
#
#
# for thresh in thresholds:
# 	# select features using threshold
# 	selection = SelectFromModel(model, threshold=thresh, prefit=True)
# 	select_X_train = selection.transform(X_train)
# 	# train model
# 	selection_model = XGBClassifier(learning_rate =0.01,
#  n_estimators=5000,
#  max_depth=4,
#  min_child_weight=6,
#  gamma=0,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  reg_alpha=0.005,
#  objective= 'binary:logistic',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27)
#
#
# 	selection_model.fit(select_X_train, y_train)
# 	# eval model
# 	select_X_test = selection.transform(X_test)
# 	predictions = selection_model.predict(select_X_test)
#
#     # y_predict = predictions
#     # y_predict[y_predict=='correct'] = 1
#     # y_predict[y_predict=='incorrect'] = 0
#     # y_predict = y_predict.reshape(y_predict.shape[0],1)
#     #
#     # trip_id = df2.iloc[:,0].values
#     # trip_id = trip_id.reshape(trip_id.shape[0],1)
#     #
#     # results = np.hstack((trip_id,y_predict))
#     # results = pd.DataFrame({'tripid': results[:,0], 'prediction': results[:,1]})
#     # # np.savetxt("results.csv", results, fmt='%i', delimiter=",")
#     # results.to_csv('results.csv',index = False)
#     #
#     #
#     # print("Completed")
# 	fbeta = fbeta_score(y_test.astype(int), predictions.astype(int), average='macro', beta=0.5)
# 	print("Thresh=%.3f, n=%d, fbeta: %.2f%%" % (thresh, select_X_train.shape[1], fbeta*100.0))




X_test = df2.iloc[:,1:].values

xgb = XGBClassifier(learning_rate =0.01,
 n_estimators=5000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)


###################################
########## Early Stopping  ########
###################################


# from sklearn.model_selection import train_test_split
#
# X_train, X_eval, y_train, y_eval = train_test_split(
#     X_train,
#     y_train,
#     test_size=0.33,
#     shuffle=True,
#     stratify=y_train,
#     random_state=6
# )
# eval_set = [(X_train, y_train), (X_eval, y_eval)]
# xgb.fit(X_train, y_train, early_stopping_rounds=6, eval_metric= "logloss", eval_set=eval_set, verbose=True)


########################################################
#########  Oversampling no improvemnt  #################
########################################################

# from collections import Counter
# from gsmote import GeometricSMOTE
#
# print('Original dataset shape %s' % Counter(y_train))
# y_train = y_train.astype('int')
# print(y_train.dtype)
# gsmote = GeometricSMOTE(random_state=1)
# X_resampled, y_resampled = gsmote.fit_resample(X_train, y_train)
# print('Resampled dataset shape %s' % Counter(y_resampled))
#
#
# xgb.fit(X_resampled, y_resampled)
#
# y_predict = xgb.predict(X_test)



xgb.fit(X_train, y_train)

y_predict = xgb.predict(X_test)


y_predict[y_predict=='correct'] = 1
y_predict[y_predict=='incorrect'] = 0
y_predict = y_predict.reshape(y_predict.shape[0],1)

trip_id = df2.iloc[:,0].values
trip_id = trip_id.reshape(trip_id.shape[0],1)

results = np.hstack((trip_id,y_predict))
results = pd.DataFrame({'tripid': results[:,0], 'prediction': results[:,1]})
results.to_csv('results.csv',index = False)


print("Completed!!!")
