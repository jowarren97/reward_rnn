import torch
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, StepLR, ConstantLR

class CustomParameterScheduler:
    def __init__(self, name, initial_value, scheduler_class, **scheduler_params):
        # Create a dummy optimizer with a single parameter for the custom value
        self.name = name
        self.dummy_param = torch.nn.Parameter(torch.tensor([initial_value]))
        self.optimizer = torch.optim.SGD([self.dummy_param], lr=initial_value)
        self.param_value = self.optimizer.param_groups[0]['lr']

        # Initialize the scheduler
        self.scheduler = scheduler_class(self.optimizer, **scheduler_params)

    def step(self):
        # Step the scheduler - it modifies the 'lr', which is our custom parameter
        self.optimizer.step()
        self.scheduler.step()

        # Retrieve the updated value
        self.param_value = self.optimizer.param_groups[0]['lr']

    def get_param(self):
        # Get the current value of the parameter
        return self.param_value
    
    
class Scheduler():
    def __init__(self):
        self.schedulers = []
        self.log = {}

    def step(self):
        # Step each scheduler
        for scheduler in self.schedulers:
            scheduler.step()

    def get_params(self):
        param_dict = {}

        for scheduler in self.schedulers:
            data = scheduler.get_param()
            param_dict[scheduler.name] = data
            self.log[scheduler.name]["data"].append(data)

        return param_dict
    
    def get_log(self):
        return self.log
    
    def reset_log(self):
        for key in self.log.keys():
            self.log[key]['data'] = []
    
class ReversalScheduler(Scheduler):
    def __init__(self, total_iters):
        super().__init__()
        # self.schedulers.append(CustomParameterScheduler('dropout', 1.0, LinearLR, start_factor=1.0, end_factor=0.0, total_iters=total_iters))
        # self.schedulers.append(CustomParameterScheduler('dropout', 1.0, StepLR, step_size=total_iters, gamma=0.5))
        self.schedulers.append(CustomParameterScheduler('dropout', 0.0, ConstantLR, factor=1.0, total_iters=1))
# scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        self.log = {'dropout' : {"data": [], "tb": True, "save": False}}

# Initialize the custom parameter scheduler
# alpha_scheduler = CustomParameterScheduler(1.0, LinearLR, start_factor=1.0, end_factor=0.0, total_iters=25)
# Example training loop
# env_scheduler = ReversalEnvironmentScheduler()

# for epoch in range(50):
#     # Training steps...
#     # ...

#     # Update the custom parameter
#     # alpha_scheduler.step()
#     # alpha = alpha_scheduler.get_param()
#     env_scheduler.step()
#     alpha = env_scheduler.get_params()
#     print(f"Epoch {epoch}, Alpha: {alpha}")
