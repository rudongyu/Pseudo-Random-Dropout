import numpy as np
import tensorflow as tf

class Simulator(object):

    def __init__(self):
        self.history = {}
        self.lambda_list = [0.8, 0.2]
        self.lower_bound = 1e-7
        self.gamma = 0.95

    def get_random_vector(self, n, p):
        res = self.history.get((n,p), 0)
        if(res):
            return res
        res = self.calc_random_vector(n, p)
        self.history[(n, p)] = res
        return res

    def calc_random_vector(self, n, p):
        epsilon = 0.5
        size = min(n/2, 64)
        cur = np.zeros(shape = size, dtype = np.uint8)
        while(epsilon > self.lower_bound):
            if(np.random.ranf()<epsilon):
                dif = np.random.choice([-1,1])
                idx = np.random.randint(0, size)
            else:
                cost = 1e6
                dif = 0
                idx = 0
                for i in xrange(size):
                    for d in [-1, 1]:
                        next = cur.copy()
                        next[i] = (next[i]+d)%n
                        e1 = abs(self.calc_prob(next)-p)
                        e2 = 0
                        for arg in range(1, size):
                            e2 += abs(self.calc_joint_prob(next, arg)-p*p)
                        new_cost = self.lambda_list[0]*e1 + self.lambda_list[1]*e2
                        if(new_cost < cost):
                            cost = new_cost
                            idx = i
                            dif = d
            cur[idx] = (cur[idx]+dif)%n
            epsilon *= self.gamma
        return cur

    def calc_prob(self, v):
        r = np.arange(1, v.shape[0]+1, dtype = np.int32)
        r = 1./r
        f1 = np.sum(v)
        f2 = np.sum(v*r)
        if(f1==0):
            return 0
        return f2/f1

    def calc_joint_prob(self, v, dif):
        r = np.arange(1, v.shape[0]+1, dtype = np.int32)
        x = ~((dif%r).astype(bool))
        r = 1./r
        f1 = np.sum(v)
        f2 = np.sum(v*r*x)
        if(f1==0):
            return 0
        return f2/f1

if __name__ == '__main__':
    sim = Simulator()
    file = open("speed", "w")
    for n in [2304, 384, 192]:
        for p in [0.7]:
            print n, p
            x = sim.get_random_vector(n, p)
            file.write(str(x)+'\n')
            print x
            print sim.calc_prob(x)
            s = np.sum(x)
            s0 = 0
            for i in range(len(x)):
                s0 += 1.0/(i+1)*x[i]
            file.write("{0}\t{1}\t{2}\n".format(n, p, s/s0))
