class EarlyStopper:
    def __init__(self, waiting=1, mind=0):
        self.waiting = waiting
        self.mind = mind
        self.c = 0
        self.min_val_loss = float('inf')

    def early_stop(self, val_loss):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.c = 0
        elif val_loss > (self.min_val_loss + self.mind):
            self.c += 1
            if self.c >= self.waiting:
                return True
        return False