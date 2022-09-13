import numpy as np
import pandas as pd
from itertools import product
from utils import get_log_beta_pdf, get_log_normal_pdf, get_log_lognormal_pdf
import progressbar

class Dina(object):

    def __init__(self, Q, X=None, max_score=1):

        self._Q = Q
        self._X = X
        self.max_score = max_score
    

    def get_student_size(self):
        return self._X.shape[0]

    def get_item_size(self):
        return self._Q.shape[0]

    def get_skill_size(self):
        return self._Q.shape[1]

    def get_eta(self, skills):
        eta = np.dot(skills, self._Q.T)
        qq = np.sum(self._Q.T * self._Q.T, axis=0)
        eta[eta < qq] = 0
        eta[eta == qq] = 1
        return eta#L*J

    def get_P_success(self, eta, slip, guess):
        P_success = (guess ** (1 - eta)) * ((1 - slip) ** eta)
        P_success[P_success <= 0] = 1e-10
        P_success[P_success >= 1] = 1 - 1e-10
        return P_success#L*J

    def get_log_Likelihood_matrix(self, P_success):
        log_P_success = np.log(P_success)
        log_P_failure = np.log(1 - P_success)
        X = self._X 
        return np.dot(log_P_success, X.T) + np.dot(log_P_failure, (1-X).T)#L*I

    def get_all_skills(self):
        skill_size = self.get_skill_size()
        return np.array(list(product([0, 1], repeat=skill_size)))

class MlDina(Dina):
    def __init__(self, guess, slip, *args, **kwargs):
        super(MlDina, self).__init__(*args, **kwargs)
        self._guess = guess
        self._slip = slip

    def get_skills_by_Ml(self):
        skills = self.get_all_skills()
        eta = self.get_eta(skills)
        P_success = self.get_P_success(eta, self._slip, self._guess)
        log_Likelihood_matrix = self.get_log_Likelihood_matrix(P_success)
        
        return skills[log_Likelihood_matrix.argmax(axis=0)]

class EmDina(Dina):
    def __init__(self, guess, slip, max_iter=100, tol=1e-5, *args, **kwargs):
        super(EmDina, self).__init__(*args, **kwargs)
        self._guess = guess
        self._slip = slip
        self._max_iter = max_iter
        self._tol = tol
        self._skills = self.get_all_skills()

    def get_posterior_matrix(self, P_success):
        
        L = P_success.shape[0]
        return np.exp(self.get_log_Likelihood_matrix(P_success) + 1.0 / L)

    def get_posterior_matrix_normalize(self, P_success):
        
        posterior_matrix = self.get_posterior_matrix(P_success)
        return posterior_matrix / np.sum(posterior_matrix, axis=0)

    def get_skill_distribution(self, posterior_matrix_normalize):
        
        return np.sum(posterior_matrix_normalize, axis=1)


    def get_init_eta_item_distribution(self, posterior_matrix_normalize):
        
        eta_item_dis_0 = np.repeat(self.get_skill_distribution(posterior_matrix_normalize), self.get_item_size()).reshape(posterior_matrix_normalize.shape[0], self.get_item_size())
        eta_item_dis_1 = eta_item_dis_0.copy()
        return eta_item_dis_0, eta_item_dis_1

    def get_eta_item_distribuiton(self, eta_item_dis_0, eta_item1_post_normalize_0, eta_item_dis_1, eta_item1_post_normalize_1, eta):
        
        eta_item_dis_0[eta == 1] = 0
        eta0_item_dis = np.sum(eta_item_dis_0, axis=0)
        eta0_item_dis[eta0_item_dis <= 0] = 1e-10
        eta_item1_post_normalize_0[eta == 1] = 0
        eta0_item1_dis = np.sum(eta_item1_post_normalize_0, axis=0)

        eta_item_dis_1[eta == 0] = 0
        eta1_item_dis = np.sum(eta_item_dis_1, axis=0)
        eta_item1_post_normalize_1[eta == 0] = 0
        eta1_item1_dis = np.sum(eta_item1_post_normalize_1, axis=0)

        return eta0_item1_dis, eta0_item_dis, eta1_item1_dis, eta1_item_dis

    def get_est_guess(self, eta0_item1_dis, eta0_item_dis):
        guess = eta0_item1_dis / eta0_item_dis
        guess[guess <= 0] = 1e-10
        return guess

    def get_est_slip(self, eta1_item1_dis, eta1_item_dis):
        slip = (eta1_item_dis - eta1_item1_dis) / eta1_item_dis
        slip[slip <= 0] = 1e-10
        return slip

    def em(self):
        skills = self.get_all_skills()
        eta = self.get_eta(skills)
        X = self._X
        max_iter = self._max_iter
        tol = self._tol
        guess = self._guess
        slip = self._slip

        count = 1

        for i in range(max_iter):
            P_success = self.get_P_success(eta, slip, guess)
            posterior_matrix_normalize = self.get_posterior_matrix_normalize(P_success)
            eta_item_dis_0, eta_item_dis_1 = self.get_init_eta_item_distribution(posterior_matrix_normalize)

            eta_item1_post_normalize_0 = np.dot(posterior_matrix_normalize, X)
            eta_item1_post_normalize_1 = eta_item1_post_normalize_0.copy()

            eta0_item1_dis, eta0_item_dis, eta1_item1_dis, eta1_item_dis\
            =self.get_eta_item_distribuiton(eta_item_dis_0, eta_item1_post_normalize_0, eta_item_dis_1, eta_item1_post_normalize_1, eta)

            guess_est = self.get_est_guess(eta0_item1_dis, eta0_item_dis)
            slip_est = self.get_est_slip(eta1_item1_dis, eta1_item_dis)

            

            if (np.max(np.abs(guess - guess_est)) < tol) and (np.max(np.abs(slip - slip_est)) < tol):
                return slip, guess

            slip = slip_est
            guess = guess_est
            count = count + 1
        #改
        return slip, guess



class McmcDina(Dina):
    def __init__(self, thin=1, burn=3000, max_iter=10000, *args, **kwargs):
        super(McmcDina, self).__init__(*args, **kwargs)
        self._burn = burn
        self._thin = thin
        self._max_iter = max_iter * thin

    def get_log_Likelihood(self, eta, slip, guess):
        X = self._X
        P_success = self.get_P_success(eta, slip, guess)
        return (X * np.log(P_success) + (1 - X) * np.log(1 - P_success))

    def init_3_parameter(self, size):
        student_size = self.get_student_size()
        item_size = self.get_item_size()
        skill_size = self.get_skill_size()

        skills = np.ones((student_size, skill_size))
        skills_list = np.zeros((size, student_size, skill_size))
        slip = np.zeros((1, item_size))
        slip_list = np.zeros((size, item_size))
        guess = np.zeros((1, item_size))
        guess_list = np.zeros((size, item_size))
        return skills, skills_list, slip, slip_list, guess, guess_list

    def get_Q_slip_and_guess(self, skills, slip, guess, next_slip, next_guess):#guess. slip的转移矩阵
        eta = self.get_eta(skills)
        log_Likelihood_matrix = self.get_log_Likelihood(eta, slip, guess)#536*20
        log_distribution = np.sum(log_Likelihood_matrix, axis=0)
        pre = log_distribution + get_log_beta_pdf(slip, guess)
        next = log_distribution + get_log_beta_pdf(next_slip, next_guess)
        Q_item_trans = np.exp(next - pre)
        Q_item_trans[Q_item_trans > 1] = 1 
        return Q_item_trans

    def update_slip_and_guess(self, skills, slip, guess):
        next_slip = np.random.uniform(slip - 0.1, slip + 0.1)
        next_guess = np.random.uniform(guess - 0.1, guess + 0.1)
        next_slip[next_slip<=0] = 1e-10
        next_slip[next_slip>=0.5] = 0.5 - 1e-10
        next_guess[next_guess<=0] = 1e-10
        next_guess[next_guess>=0.5] = 0.5 - 1e-10
        Q_item_trans = self.get_Q_slip_and_guess(skills, slip, guess, next_slip, next_guess)
        u = np.random.uniform(0, 1, Q_item_trans.shape)
        slip[Q_item_trans >= u] = next_slip[Q_item_trans >= u]
        guess[Q_item_trans >= u] = next_guess[Q_item_trans >= u]
        return slip, guess



    def get_Q_skills(self, slip, guess, skills, next_skills):
        eta = self.get_eta(skills)
        Likelihood_matrix = self.get_log_Likelihood(eta, slip, guess)
        pre = np.sum(Likelihood_matrix, axis=1)

        eta = self.get_eta(next_skills)
        Likelihood_matrix = self.get_log_Likelihood(eta, slip, guess)
        next = np.sum(Likelihood_matrix, axis=1)

        Q_skills_trans = np.exp(next-pre)
        Q_skills_trans[Q_skills_trans>1] = 1 
        return Q_skills_trans

    def update_skills(self, skills, slip, guess):
        next_skills = np.random.binomial(1, 0.5, skills.shape)
        Q_skills = self.get_Q_skills(slip, guess, skills, next_skills)
        u_skills = np.random.uniform(0, 1, Q_skills.shape)
        skills[Q_skills >= u_skills] = next_skills[Q_skills >= u_skills]
        return skills#返回一个I*L的skills矩阵

    def mcmc(self):
        size = self._max_iter
        skills, skills_list, slip, slip_list, guess, guess_list = self.init_3_parameter(size)

        for i in range(size):
            skills = self.update_skills(skills, slip, guess)
            slip, guess = self.update_slip_and_guess(skills, slip, guess)
            skills_list[i] = skills
            slip_list[i] = slip
            guess_list[i] = guess
            print(i)
        est_skills = np.mean(skills_list[self._burn::self._thin], axis=0)#np.mean(skills_list[::], axis=0):把每个skills矩阵加起来再处以矩阵的个数
        est_slip = np.mean(slip_list[self._burn::self._thin], axis=0)#thin?
        est_guess = np.mean(guess_list[self._burn::self._thin], axis=0)
        return est_skills, est_slip, est_guess


class McmcHoDina(Dina):
    def __init__(self, thin=1, burn=5000, max_iter=10000, *args, **kwargs):
        super(McmcHoDina, self).__init__(*args, **kwargs)
        self._burn = burn
        self._thin = thin
        self._max_iter = max_iter * thin

    def get_log_Likelihood(self, eta, slip, guess):
        X = self._X
        P_success = self.get_P_success(eta, slip, guess)
        return (X * np.log(P_success) + (1 - X) * np.log(1 - P_success))

    def init_parameters(self):
        size = self._max_iter
        student_size = self.get_student_size()
        skills_size = self.get_skill_size()
        
        theta = np.zeros((student_size, 1))
        theta_list = np.zeros((size, student_size, 1))
        lambda0 = np.zeros(skills_size)
        lambda0_list = np.zeros((size, skills_size))
        lambda1 = np.ones(skills_size)
        lambda1_list = np.zeros((size, skills_size))
        
        return theta, theta_list, lambda0, lambda0_list, lambda1, lambda1_list

    def init_3_parameter(self, size):
        student_size = self.get_student_size()
        item_size = self.get_item_size()
        skill_size = self.get_skill_size()

        skills = np.ones((student_size, skill_size))
        skills_list = np.zeros((size, student_size, skill_size))
        slip = np.zeros((1, item_size))+0.3
        slip_list = np.zeros((size, item_size))
        guess = np.zeros((1, item_size))+0.3
        guess_list = np.zeros((size, item_size))
        return skills, skills_list, slip, slip_list, guess, guess_list

    def get_P_skills(self, theta, lambda0, lambda1):
        #lambda的原因20应转为8
        exp_P = np.exp(lambda0 + theta * lambda1)
        return exp_P / (1.0 + exp_P)

    def get_skills_pdf(self, skills, theta, lambda0, lambda1):
        P_skills = self.get_P_skills(theta, lambda0, lambda1)
        P_skills[P_skills <= 0] = 1e-10
        P_skills[P_skills >= 1] = 1 - 1e-10
        return skills * np.log(P_skills) + (1 - skills) * (np.log(1 - P_skills))

    def get_skills_Q(self, skills, next_skills, slip, guess, theta, lambda0, lambda1):
        eta = self.get_eta(skills)
        pre = np.sum(self.get_log_Likelihood(eta, slip, guess), axis=1) +\
              np.sum(self.get_skills_pdf(skills, theta, lambda0, lambda1), axis=1)

        next_eta = self.get_eta(next_skills)
        next = np.sum(self.get_log_Likelihood(next_eta, slip, guess), axis=1) +\
               np.sum(self.get_skills_pdf(next_skills, theta, lambda0, lambda1), axis=1)
        skills_Q = np.exp(next - pre)
        skills_Q[skills_Q > 1] = 1
        return skills_Q

    def get_theta_Q(self, skills, theta, next_theta, lambda0, lambda1):
        pre = np.sum(self.get_skills_pdf(skills, theta, lambda0, lambda1), axis=1) + get_log_normal_pdf(theta)[:, 0]
        next = np.sum(self.get_skills_pdf(skills, next_theta, lambda0, lambda1), axis=1) + get_log_normal_pdf(next_theta)[:, 0]
        theta_Q = np.exp(next - pre)
        theta_Q[theta_Q > 1] = 1
        return theta_Q

    def get_lambda_Q(self, skills, theta, lambda0, next_lambda0, lambda1, next_lambda1):
        pre = np.sum(self.get_skills_pdf(skills, theta, lambda0, lambda1), axis=0) +\
              get_log_normal_pdf(lambda0) + get_log_normal_pdf(lambda1)
        next = np.sum(self.get_skills_pdf(skills, theta, next_lambda0, next_lambda1), axis=0) +\
              get_log_normal_pdf(next_lambda0) + get_log_normal_pdf(next_lambda1)
        lambda_Q = np.exp(next - pre)
        lambda_Q[lambda_Q > 1] = 1
        return lambda_Q

    def update_skills(self, skills, slip, guess, theta, lambda0, lambda1):
        next_skills = np.random.binomial(1, 0.5, skills.shape)
        skills_Q = self.get_skills_Q(skills, next_skills, slip, guess, theta, lambda0, lambda1)
        skills_mu = np.random.uniform(0, 1, skills_Q.shape)
        skills[skills_Q >= skills_mu] = next_skills[skills_Q >= skills_mu]
        return skills

    def update_theta(self, skills, theta,  lambda0, lambda1):
        next_theta = np.random.normal(theta, 0.1)#shape?
        theta_Q = self.get_theta_Q(skills, theta, next_theta, lambda0, lambda1)
        theta_mu = np.random.uniform(0, 1, theta_Q.shape)
        theta[theta_Q > theta_mu] = next_theta[theta_Q > theta_mu]
        return theta

    def update_lambda(self, skills, theta, lambda0, lambda1):
        next_lambda0 = np.random.uniform(lambda0 - 0.3, lambda0 + 0.3)
        next_lambda1 = np.random.uniform(lambda1 - 0.3, lambda1 + 0.3)
        next_lambda1[next_lambda1 <= 0] = 1e-10
        next_lambda1[next_lambda1 > 4] = 4
        lambda_Q = self.get_lambda_Q(skills, theta, lambda0, next_lambda0, lambda1, next_lambda1)
        lambda_mu = np.random.uniform(0, 1, lambda_Q.shape)
        lambda0[lambda_Q > lambda_mu] = lambda0[lambda_Q > lambda_mu]
        lambda1[lambda_Q > lambda_mu] = lambda1[lambda_Q > lambda_mu]
        return lambda0, lambda1


    def get_Q_slip_and_guess(self, skills, slip, guess, next_slip, next_guess):
        eta = self.get_eta(skills)
        pre = np.sum(self.get_log_Likelihood(eta, slip, guess), axis=0) + get_log_beta_pdf(slip, guess)
        next = np.sum(self.get_log_Likelihood(eta, next_slip, next_guess), axis=0) + get_log_beta_pdf(next_slip, next_guess)
        Q_item_trans = np.exp(next - pre)
        Q_item_trans[Q_item_trans > 1] = 1 
        return Q_item_trans

    def update_slip_and_guess(self, skills, slip, guess):
        next_slip = np.random.uniform(slip - 0.1, slip + 0.1)
        next_guess = np.random.uniform(guess - 0.1, guess + 0.1)
        next_slip[next_slip<=0] = 1e-10
        next_slip[next_slip>=0.6] = 0.6 - 1e-10
        next_guess[next_guess<=0] = 1e-10
        next_guess[next_guess>=0.6] = 0.6 - 1e-10
        Q_item_trans = self.get_Q_slip_and_guess(skills, slip, guess, next_slip, next_guess)
        u = np.random.uniform(0, 1, Q_item_trans.shape)
        slip[Q_item_trans >= u] = next_slip[Q_item_trans >= u]
        guess[Q_item_trans >= u] = next_guess[Q_item_trans >= u]
        return slip, guess



    
    def mcmc(self):
        #bar = progressbar.ProgressBar()
        theta, theta_list, lambda0, lambda0_list, lambda1, lambda1_list = self.init_parameters()
        skills, skills_list, slip, slip_list, guess, guess_list = self.init_3_parameter(self._max_iter)
        #for i in bar(range(self._max_iter)):
        for i in range(self._max_iter):

            theta = self.update_theta(skills, theta,  lambda0, lambda1)
            theta_list[i] = theta

            lambda0, lambda1 = self.update_lambda(skills, theta, lambda0, lambda1)
            lambda0_list[i] = lambda0
            lambda1_list[i] = lambda1

            skills = self.update_skills(skills, slip, guess, theta, lambda0, lambda1)
            skills_list[i] = skills

            slip, guess = self.update_slip_and_guess(skills, slip, guess)
            slip_list[i] = slip
            guess_list[i] = guess

        lambda0_est = np.mean(lambda0_list[self._burn::self._thin], axis=0)
        lambda1_est = np.mean(lambda1_list[self._burn::self._thin], axis=0)
        theta_est = np.mean(theta_list[self._burn::self._thin], axis=0)
        skills_est = np.mean(skills_list[self._burn::self._thin], axis=0)
        slip_est = np.mean(slip_list[self._burn::self._thin], axis=0)
        guess_est = np.mean(guess_list[self._burn::self._thin], axis=0)

        return lambda0_est, lambda1_est, theta_est, skills_est, slip_est, guess_est



class McmcPHoDina(Dina):
    def __init__(self, thin=1, burn=5000, max_iter=10000, *args, **kwargs):
        super(McmcPHoDina, self).__init__(*args, **kwargs)
        self._burn = burn
        self._thin = thin
        self._max_iter = max_iter * thin

    def get_log_Likelihood(self, eta, slip, guess):
        X = self._X
        P_success = self.get_P_success(eta, slip, guess)
        return (X * np.log(P_success) + (1 - X) * np.log(1 - P_success))

    def init_latent_parameters(self):
        size = self._max_iter
        student_size = self.get_student_size()
        skills_size = self.get_skill_size()
        
        theta = np.zeros((student_size, 1))
        theta_list = np.zeros((size, student_size, 1))
        lambda0 = np.zeros(skills_size)
        lambda0_list = np.zeros((size, skills_size))
        lambda1 = np.ones(skills_size)
        lambda1_list = np.zeros((size, skills_size))
        
        return theta, theta_list, lambda0, lambda0_list, lambda1, lambda1_list

    def init_skills(self, size):
        student_size = self.get_student_size()
        skill_size = self.get_skill_size()
        skills = np.ones((student_size, skill_size))
        skills_list = np.zeros((size, student_size, skill_size))
        return skills, skills_list
    
    def init_slip_guess(self, size):
        item_size = self.get_item_size()
        slip = np.zeros((1, item_size))+0.3
        slip_list = np.zeros((size, item_size))
        guess = np.zeros((1, item_size))+0.3
        guess_list = np.zeros((size, item_size))
        return slip, slip_list, guess, guess_list

    def get_P_skills(self, theta, lambda0, lambda1):
        #lambda的原因20应转为8
        exp_P = np.exp(lambda0 + theta * lambda1)
        return exp_P / (1.0 + exp_P)

    def get_skills_pdf(self, skills, theta, lambda0, lambda1):
        P_skills = self.get_P_skills(theta, lambda0, lambda1)
        P_skills[P_skills <= 0] = 1e-10
        P_skills[P_skills >= 1] = 1 - 1e-10
        return skills * np.log(P_skills) + (1 - skills) * (np.log(1 - P_skills))

    def get_skills_Q(self, skills, next_skills, slip, guess, theta, lambda0, lambda1):
        eta = self.get_eta(skills)
        pre = np.sum(self.get_log_Likelihood(eta, slip, guess), axis=1) +\
              np.sum(self.get_skills_pdf(skills, theta, lambda0, lambda1), axis=1)

        next_eta = self.get_eta(next_skills)
        next = np.sum(self.get_log_Likelihood(next_eta, slip, guess), axis=1) +\
               np.sum(self.get_skills_pdf(next_skills, theta, lambda0, lambda1), axis=1)
        skills_Q = np.exp(next - pre)
        skills_Q[skills_Q > 1] = 1
        return skills_Q

    def get_theta_Q(self, skills, theta, next_theta, lambda0, lambda1):
        pre = np.sum(self.get_skills_pdf(skills, theta, lambda0, lambda1), axis=1) + get_log_normal_pdf(theta)[:, 0]
        next = np.sum(self.get_skills_pdf(skills, next_theta, lambda0, lambda1), axis=1) + get_log_normal_pdf(next_theta)[:, 0]
        theta_Q = np.exp(next - pre)
        theta_Q[theta_Q > 1] = 1
        return theta_Q

    def get_lambda_Q(self, skills, theta, lambda0, next_lambda0, lambda1, next_lambda1):
        pre = np.sum(self.get_skills_pdf(skills, theta, lambda0, lambda1), axis=0) +\
              get_log_normal_pdf(lambda0) + get_log_normal_pdf(lambda1)
        next = np.sum(self.get_skills_pdf(skills, theta, next_lambda0, next_lambda1), axis=0) +\
              get_log_normal_pdf(next_lambda0) + get_log_normal_pdf(next_lambda1)
        lambda_Q = np.exp(next - pre)
        lambda_Q[lambda_Q > 1] = 1
        return lambda_Q

    def update_skills(self, skills, slip, guess, theta, lambda0, lambda1):
        next_skills = np.random.binomial(1, 0.5, skills.shape)
        skills_Q = self.get_skills_Q(skills, next_skills, slip, guess, theta, lambda0, lambda1)
        skills_mu = np.random.uniform(0, 1, skills_Q.shape)
        skills[skills_Q >= skills_mu] = next_skills[skills_Q >= skills_mu]
        return skills

    def update_theta(self, skills, theta,  lambda0, lambda1):
        next_theta = np.random.normal(theta, 0.1)#shape?
        theta_Q = self.get_theta_Q(skills, theta, next_theta, lambda0, lambda1)
        theta_mu = np.random.uniform(0, 1, theta_Q.shape)
        theta[theta_Q > theta_mu] = next_theta[theta_Q > theta_mu]
        return theta

    def update_lambda(self, skills, theta, lambda0, lambda1):
        next_lambda0 = np.random.uniform(lambda0 - 0.3, lambda0 + 0.3)
        next_lambda1 = np.random.uniform(lambda1 - 0.3, lambda1 + 0.3)
        next_lambda1[next_lambda1 <= 0] = 1e-10
        next_lambda1[next_lambda1 > 4] = 4
        lambda_Q = self.get_lambda_Q(skills, theta, lambda0, next_lambda0, lambda1, next_lambda1)
        lambda_mu = np.random.uniform(0, 1, lambda_Q.shape)
        lambda0[lambda_Q > lambda_mu] = lambda0[lambda_Q > lambda_mu]
        lambda1[lambda_Q > lambda_mu] = lambda1[lambda_Q > lambda_mu]
        return lambda0, lambda1


    def get_Q_slip_and_guess(self, skills, slip, guess, next_slip, next_guess):#guess. slip的转移矩阵
        eta = self.get_eta(skills)
        pre = np.sum(self.get_log_Likelihood(eta, slip, guess), axis=0) + get_log_beta_pdf(slip, guess)
        next = np.sum(self.get_log_Likelihood(eta, next_slip, next_guess), axis=0) + get_log_beta_pdf(next_slip, next_guess)
        Q_item_trans = np.exp(next - pre)
        Q_item_trans[Q_item_trans > 1] = 1 
        return Q_item_trans

    def update_slip_and_guess(self, skills, slip, guess):
        next_slip = np.random.uniform(slip - 0.1, slip + 0.1)
        next_guess = np.random.uniform(guess - 0.1, guess + 0.1)
        next_slip[next_slip<=0] = 1e-10
        next_slip[next_slip>=0.6] = 0.6 - 1e-10
        next_guess[next_guess<=0] = 1e-10
        next_guess[next_guess>=0.6] = 0.6 - 1e-10
        Q_item_trans = self.get_Q_slip_and_guess(skills, slip, guess, next_slip, next_guess)
        u = np.random.uniform(0, 1, Q_item_trans.shape)
        slip[Q_item_trans >= u] = next_slip[Q_item_trans >= u]
        guess[Q_item_trans >= u] = next_guess[Q_item_trans >= u]
        return slip, guess



    
    def mcmc(self):
        #bar = progressbar.ProgressBar()
        theta, theta_list, lambda0, lambda0_list, lambda1, lambda1_list = self.init_latent_parameters()
        skills, skills_list= self.init_skills(self._max_iter)
        
        #for i in bar(range(self._max_iter)):
        for i in range(self._max_iter):

            theta = self.update_theta(skills, theta,  lambda0, lambda1)
            theta_list[i] = theta

            lambda0, lambda1 = self.update_lambda(skills, theta, lambda0, lambda1)
            lambda0_list[i] = lambda0
            lambda1_list[i] = lambda1

            skills = self.update_skills(skills, slip, guess, theta, lambda0, lambda1)
            skills_list[i] = skills

            for t in range(self.max_score):
                
                slip, slip_list, guess, guess_list  = self.init_slip_guess(self._max_iter)
                slip, guess = self.update_slip_and_guess(skills, slip, guess)
                slip_list[i] = slip
                guess_list[i] = guess

        lambda0_est = np.mean(lambda0_list[self._burn::self._thin], axis=0)
        lambda1_est = np.mean(lambda1_list[self._burn::self._thin], axis=0)
        theta_est = np.mean(theta_list[self._burn::self._thin], axis=0)
        skills_est = np.mean(skills_list[self._burn::self._thin], axis=0)
        slip_est = np.mean(slip_list[self._burn::self._thin], axis=0)
        guess_est = np.mean(guess_list[self._burn::self._thin], axis=0)

        return lambda0_est, lambda1_est, theta_est, skills_est, slip_est, guess_est





