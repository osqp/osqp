class Results(object):
    '''
    Results class from QP solution
    '''
    def __init__(self, status, obj_val, x, y, run_time, iter):
        self.status = status
        self.obj_val = obj_val
        self.x = x
        self.y = y
        self.run_time = run_time
        self.iter = iter
