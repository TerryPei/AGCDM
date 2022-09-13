import numpy as np
import pandas as pd
from utils import r4beta
import Dina

Q = np.array(pd.read_csv('Q.txt', header=None, sep='\t'))
X = np.array(pd.read_csv('X.txt', header=None, sep='\t'))
dina = Dina.Dina(Q, X)
student_size = dina.get_student_size()
item_size = dina.get_item_size()
skill_size = dina.get_skill_size()
#guess = r4beta(1, 2, 0, 0.6, (1, item_size))
#slip = r4beta(1, 2, 0, 0.6, (1, item_size))
guess = np.zeros(item_size)
slip = np.zeros(item_size)
guess[:] = slip[:] = 0.01
max_iter = 100
tol = 1e-5

emDina = Dina.EmDina(guess, slip, max_iter, tol, Q, X)
slip_est, guess_est = emDina.em()

print(np.mean(np.abs(slip_est - slip)))
print(np.mean(np.abs(guess_est - guess)))

dina_est = Dina.MlDina(guess_est, slip_est, Q, X)
skills_est = dina_est.get_skills_by_Ml()







#print(skills_est)

'''

print("The number of student: {}".format(student_size))
print("The number of question: {}".format(item_size))
print("The number of skill: {}".format(skill_size))
eta = Dina.get_eta(skills)

P_success = Dina.get_P_success(eta, slip, guess)


print("Eta_Matrix(Whether student can solve the item j by skill k): \n{}".format(eta))
print("Probability of answering correctly: \n{}".format(P_success))
print("eta.shape: \n{}".format(eta.shape))
print("P_success.shape: \n{}".format(P_success.shape))
'''
