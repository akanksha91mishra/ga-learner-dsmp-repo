# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'
data = np.genfromtxt(path, delimiter=",", skip_header=1)

#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]

#Code starts here
census = np.concatenate([data,new_record])
print(census)



# --------------
#Code starts here
age = census[:,0:1]
max_age = np.max(age)
min_age = np.min(age)
age_mean = np.mean(age)
age_std = np.std(age)
print(max_age)
print(min_age)
print(age_mean)
print(age_std)


# --------------
#Code starts here
race_0 = census[census[:,2]==0]
race_1 = census[census[:,2]==1]
race_2 = census[census[:,2]==2]
race_3 = census[census[:,2]==3]
race_4 = census[census[:,2]==4]

len_0 = len(race_0)
len_1 = len(race_1)
len_2 = len(race_2)
len_3 = len(race_3)
len_4 = len(race_4)
citizens = [len_0, len_1, len_2, len_3, len_4]
#print(citizens)
minority_race = citizens.index(min(citizens))
print(minority_race)



# --------------
#Code starts here



senior_citizens = census[census[:,0]>60]
senior_citizens = np.array([x for x in census[:,0:1] if x>60])
#working_hours_sum = sum([x for x in census[:,6:7] if x[0] > 60])
working_hours_sum = sum(census[census[:,0]>60][:,6])
print(working_hours_sum)
senior_citizens_len = len(senior_citizens)

avg_working_hours = working_hours_sum/senior_citizens_len

print(avg_working_hours)




# --------------
#Code starts here
high = census[census[:,1]>10]
low = census[census[:,1]<=10]
#print(high)
#print(low)
avg_pay_high = sum(census[census[:,1]>10][:,7])/len(high)
print(avg_pay_high)
avg_pay_low = sum(census[census[:,1]<=10][:,7])/len(low)
print(avg_pay_low)



