import torch
import torch.nn as nn
import numpy as np


class BaseActor(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[256, 256]):
        super(BaseActor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        # self.dropout = dropout
        # Build layers
        self.layers = self.build_layers()
        self.output_layer = nn.Linear(self.hidden_sizes[-1], self.output_size)

    def build_layers(self):
        prev_size = self.input_size
        layers = nn.ModuleList()
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        return layers

    def get_embedding(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_embedding_and_logits(self, x):
        embedding = self.get_embedding(x)
        logits = self.output_layer(embedding)
        return embedding, logits

    def forward(self, x):
        embedding = self.get_embedding(x)
        logits = self.output_layer(embedding)
        return logits


class GoalConditionedActor(nn.Module):
    def __init__(self, 
                 obs_size, 
                 condition_size, 
                 action_size,
                 use_teacher=False, 
                 student_hidden_sizes=[256, 256], 
                 teacher_hidden_sizes=[256, 256]):
        super(GoalConditionedActor, self).__init__()
        self.student_actor = BaseActor(
            input_size=obs_size + condition_size,
            output_size=action_size,
            hidden_sizes=student_hidden_sizes,
        )

        self.teacher_actor = None # Optional teacher actor
        if use_teacher:
            self.teacher_actor = BaseActor(
                input_size=2*obs_size + condition_size, # teacher knows the exact state to be reached
                output_size=action_size,
                hidden_sizes=teacher_hidden_sizes,
            )

    def get_student_embedding_and_logits(self, obs, condition):
        x = torch.cat([obs, condition], dim=-1)
        return self.student_actor.get_embedding_and_logits(x)

    def get_teacher_embedding_and_logits(self, obs, final_obs, condition):
        if self.teacher_actor is not None:
            x = torch.cat([obs, final_obs, condition], dim=-1)
            return self.teacher_actor.get_embedding_and_logits(x)
        else:
            raise ValueError("Teacher actor is not available.")
        
    def forward(self, x):
        # When deploying in the environment, only the student actor is used
        return self.student_actor(x)