import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import pandas as pd
import numpy as np
import os
# Constants
num_systems = 4
num_continuous_controls = num_systems * 9 #冷机冷冻水出水温度，冷却水进水温度,冷塔频率1,2,3,4，冷却塔冷却水出水温度,冷却泵功率,冷冻泵功率

num_features = 29*4


num_towers_per_system=4
# For each system:
num_chiller_actions = 2  # On or off
num_plate_exchanger_actions = 2  # On or off
num_cooling_tower_combinations = 2 * num_towers_per_system  # Each can be on or off

# The action to turn the entire system on or off (master switch)
num_system_off_action = 2

# For num_systems systems, the discrete action space is:
num_discrete_controls_per_system = num_chiller_actions + num_plate_exchanger_actions + num_cooling_tower_combinations
num_discrete_controls = ((num_discrete_controls_per_system+num_system_off_action) * num_systems) 

print('num_continuous_controls: {} , num_discrete_controls: {}'.format(num_continuous_controls,num_discrete_controls))

max_time=None

# Hyperparameters
learning_rate = 0.01
gamma = 0.8  # Discount factor for rewards

# Policy network
class PolicyNetwork(nn.Module):
    def __init__(self, num_features, num_continuous_controls, num_discrete_controls):
        super(PolicyNetwork, self).__init__()
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Branch for discrete controls
        self.discrete_head = nn.Sequential(
            nn.Linear(64, num_discrete_controls)  # Probabilities for discrete actions
        )
        
        
        # Branch for continuous controls
        self.continuous_head = nn.Sequential(
            nn.Linear(64, num_continuous_controls * 2)  # Mean and std dev for each control
        )
        
    def forward(self, x):
        x = self.shared_layers(x)
    
        # Discrete actions output
        discrete_logits = self.discrete_head(x)
        
        # Continuous actions output
        continuous_actions = self.continuous_head(x)
        means = continuous_actions[:, :num_continuous_controls]
        std_devs = torch.clamp(continuous_actions[:, num_continuous_controls:], min=1e-3)
        
        return means, std_devs, discrete_logits

# Instantiate the policy network and the optimizer
policy = PolicyNetwork(num_features, num_continuous_controls, num_discrete_controls)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# Function to load model and optimizer states
def load_model_and_optimizer(model, optimizer, filename="model_checkpoint.pth"):
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded checkpoint from:", filename)
    else:
        print("No checkpoint found at:", filename)

load_model_and_optimizer(policy, optimizer, filename="/Users/zhiranbai/Documents/GitHub/Chiller-plate-optimizatoin/model_checkpoint.pth")



#给current_state normalize feature
def normalize(data):
    """
    Normalize a list of features with zero-padding handling.

    Parameters:
        data (list): The list of features to normalize, possibly zero-padded.
        feature_mins (list): The minimum values for each feature across all data.
        feature_maxs (list): The maximum values for each feature across all data.
    
    Returns:
        list: The normalized list of features.
    """
    feature_mins = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0] * 4  # Repeat pattern for all systems
    feature_maxs = [22.5, 26.2, 28.8, 36.1, 35.4, 39.4, 27.5, 30, 35, 39, 72.8, 6327, 43.6, 563, 49.7, 136, 52, 52 ,52,52,81.3,81.3,81.3,81.3,1300,1300,56.4,99,55] * 4
    normalized_data = []
    for i, value in enumerate(data):
        value=float(value)
        if value == 0:  # Assuming 0 is only used for zero-padding
            normalized_data.append(0)
        else:
            # Apply Min-Max normalization
            min_val = feature_mins[i]
            max_val = feature_maxs[i]
            normalized_val = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
            normalized_data.append(normalized_val)
    
    return np.array(normalized_data)

def strlist2float(stringlist):
    float_list = []
    for item in stringlist:
        try:
            float_item = float(item)
        except ValueError:
            print(f"Warning: '{item}' is not a valid float.")
            # float_item = default_value  # Use the default value
        float_list.append(float_item)
    return float_list
        
#拿到一个系统的数据(helper function)
def get_system_data(df,system_id):
    try:
        # This function would fetch or receive the current operational data for a given system
        columns=['冷机冷冻回水温度', '冷机冷冻出水温度',
            '板换冷冻回水温度', '板换冷冻出水温度', '冷机冷却回水温度', '冷机冷却出水温度', '板换冷却回水温度', '板换冷却出水温度',
            '冷塔出水温度', '冷塔回水温度', '冷机负载率', '冷机功率', '冷冻水泵频率', '冷冻水泵功率', '冷却水泵频率',
            '冷却水泵功率', '冷塔频率', '冷塔功率', '冷冻水流量', '冷却水流量','室外干球', '室外湿度', '室外湿球']    #29

        # Construct the system identifier string
        target_identifier = f"{system_id}#系统"
        
        # Find the row with the matching 'System Identifier'
        row = df[df['System Identifier'] == target_identifier]
  
        mode= row.iloc[0]['运行模式']


        if not row.empty:
            data = row[columns].iloc[0].to_list()

            # Define the required number of sub-items for indices 17 and 18
            required_count = 4

            # Process the list to handle '/' separated values and validate specific indices
            processed_data = []
            split_values_16 = None
            split_values_17 = None

            for index, item in enumerate(data):
                if isinstance(item, str) and '/' in item:
                    sub_items = item.split('/')

                    # If indices 17 or 18, check number of sub-items and adjust
                    if index == 16 or index == 17:
                        if len(sub_items) < required_count:
                            # Add zeros if there are not enough sub-items
                            sub_items.extend(['0'] * (required_count - len(sub_items)))
                        elif len(sub_items) > required_count:
                            # Raise an exception if there are more sub-items than expected
                            raise ValueError(f"Index {index} has more sub-values than expected ({required_count}).")

                        # Store the results to compare later
                        if index == 16:
                            split_values_16 = sub_items
                        else:
                            split_values_17 = sub_items

                    # Append or extend the processed data list
                    
                    processed_data.extend(strlist2float(sub_items))
                else:
                    if np.isnan(item):
                        item=float(0)
                    # Append non-split items directly
                    processed_data.append(float(item))

            # After both are processed, ensure they have the same number of items
            if split_values_16 and split_values_17 and len(split_values_16) != len(split_values_17):
                raise ValueError("The number of items at index 17 and 18 do not match after processing.")
            return processed_data,mode
        else:
            # Return None if no matching row is found
            return None
    except ValueError as e:
        print("Error:", e)
        return None
    
    


def is_system_running(latest_df,system_id):
    
    identifiers = []

    # Iterating through the rows and checking the '运行模式'
    for index, row in latest_df.iterrows():
        if row['运行模式'] is not None:  # Check if '运行模式' is not None
            identifiers.append(row['System Identifier'])  # Add the 'System Identifier' to the list
    
    target_identifier = f"{system_id}#系统"
    
    # Check if the target identifier exists in the 'System Identifier' column
    return target_identifier in latest_df['System Identifier'].values

def cooling_demand_calculation(system_data,mode):
    if mode =='冷机模式':
        
        冷冻水进水流量 = float(system_data[24])
        冷机冷冻水进水温度 = float(system_data[0])
        冷机冷冻水出水温度 = float(system_data[1])
        水比热熔=4.186*1000/3600
        return float(冷冻水进水流量 * 水比热熔 * (冷机冷冻水进水温度 - 冷机冷冻水出水温度))
    
    elif mode =='混合模式':
        板换冷冻水进水温度=float(system_data[2])
        冷机冷冻水出水温度=float(system_data[1])
        冷冻水进水流量=float(system_data[24])
        水比热熔=4.186*1000/3600
        return float((板换冷冻水进水温度-冷机冷冻水出水温度)*冷冻水进水流量*水比热熔)
        
    elif mode =='板换模式':
        板换冷冻水进水温度=float(system_data[2])
        板换冷冻水出水温度=float(system_data[3])
        冷冻水进水流量=float(system_data[24])
        水比热熔=4.186*1000/3600
        return float((板换冷冻水进水温度-板换冷冻水出水温度)*冷冻水进水流量*水比热熔)

#拿到系统现在状态
def get_real_system_state():
    df = pd.read_excel("/Users/zhiranbai/Downloads/工作/数据中心AI/2冷源流量补值代码-3种模式/补充混合+板换+冷机水流量.xlsx")
    
    global max_time
    max_time = df['MM-DD hour'].max()

    latest_time_rows = df[df['MM-DD hour'] == max_time]    
        
    current_state = []
    cooling_demand=0
    for system_id in range(1, num_systems + 1):
        if is_system_running(latest_time_rows,system_id):
            
            
            
            system_data = None
            while system_data is None:
                system_data,mode= get_system_data(latest_time_rows,system_id)
                current_state.extend(system_data)
                cooling_demand += cooling_demand_calculation(system_data,mode)
                print("system_data: {}" .format(system_data))
                if system_data is None:
                    input("Please correct the data and press Enter to retry...")
            # print("Data processed successfully:", system_data)

            
            
            
        else:
            system_data = [0] * 29
            current_state.extend(system_data)

    current_state = np.array(current_state) # Apply normalization or standardization
    
    
    return current_state,cooling_demand


def scale_continuous_actions(actions):
    
    min_values=[0,0,0,0,0,0,0,0,0]*4  
    max_values=[26.2,35.4,52,52,52,52,35,49.7,43.6]*4 
    continuous_name=['系统1:冷机冷冻水出水温度','系统1:冷却水进水温度','系统1:冷塔频率1','系统1:冷塔频率2','系统1:冷塔频率3','系统1:冷塔频率4','系统1:冷却塔冷却水出水温度','系统1:冷却泵频率','系统1:冷冻泵频率','系统2:冷机冷冻水出水温度','系统2:冷却水进水温度','系统2:冷塔频率1','系统2:冷塔频率2','系统2:冷塔频率3','系统2:冷塔频率4','系统2:冷却塔冷却水出水温度','系统2:冷却泵频率','系统2:冷冻泵频率','系统3:冷机冷冻水出水温度','系统3:冷却水进水温度','系统3:冷塔频率1','系统3:冷塔频率2','系统3:冷塔频率3','系统3:冷塔频率4','系统3:冷却塔冷却水出水温度','系统3:冷却泵频率','系统3:冷冻泵频率','系统4:冷机冷冻水出水温度','系统4:冷却水进水温度','系统4:冷塔频率1','系统4:冷塔频率2','系统4:冷塔频率3','系统4:冷塔频率4','系统4:冷却塔冷却水出水温度','系统4:冷却泵频率','系统4:冷冻泵频率']
    scaled_actions = []
    if isinstance(actions, torch.Tensor):
        actions = actions.squeeze()
    for action, min_value, max_value, action_name in zip(actions, min_values, max_values,continuous_name):
        scaled_action = action.item() * (max_value - min_value) + min_value
        scaled_actions.append((action_name,scaled_action))
        
        
    return scaled_actions

def apply_discrete_action(action):
    """ Apply a discrete action based on the output probabilities. """
    category_List = ['System 1 Chiller: OFF', 'System 1 Chiller: ON', 'System 1 Plate Exchanger: OFF', 'System 1 Plate Exchanger: ON', 'System 1 Cooling Tower 1: OFF', 'System 1 Cooling Tower 1: ON', 'System 1 Cooling Tower 2: OFF', 'System 1 Cooling Tower 2: ON', 'System 1 Cooling Tower 3: OFF', 'System 1 Cooling Tower 3: ON', 'System 1 Cooling Tower 4: OFF', 'System 1 Cooling Tower 4: ON', 'System 1 whole: OFF','System 1 whole: ON','System 2 Chiller: OFF', 'System 2 Chiller: ON', 'System 2 Plate Exchanger: OFF', 'System 2 Plate Exchanger: ON', 'System 2 Cooling Tower 1: OFF', 'System 2 Cooling Tower 1: ON', 'System 2 Cooling Tower 2: OFF', 'System 2 Cooling Tower 2: ON', 'System 2 Cooling Tower 3: OFF', 'System 2 Cooling Tower 3: ON', 'System 2 Cooling Tower 4: OFF', 'System 2 Cooling Tower 4: ON','System 2 whole: OFF','System 2 whole: ON','System 3 Chiller: OFF', 'System 3 Chiller: ON', 'System 3 Plate Exchanger: OFF', 'System 3 Plate Exchanger: ON', 'System 3 Cooling Tower 1: OFF', 'System 3 Cooling Tower 1: ON', 'System 3 Cooling Tower 2: OFF', 'System 3 Cooling Tower 2: ON', 'System 3 Cooling Tower 3: OFF', 'System 3 Cooling Tower 3: ON', 'System 3 Cooling Tower 4: OFF', 'System 3 Cooling Tower 4: ON','System 3 whole: OFF','System 3 whole: ON','System 4 Chiller: OFF', 'System 4 Chiller: ON', 'System 4 Plate Exchanger: OFF', 'System 4 Plate Exchanger: ON', 'System 4 Cooling Tower 1: OFF', 'System 4 Cooling Tower 1: ON', 'System 4 Cooling Tower 2: OFF', 'System 4 Cooling Tower 2: ON', 'System 4 Cooling Tower 3: OFF', 'System 4 Cooling Tower 3: ON', 'System 4 Cooling Tower 4: OFF', 'System 4 Cooling Tower 4: ON', 'System 4 whole: OFF','System 4 whole: ON'] 
    chosen_action= category_List[action]
    return chosen_action



def save_model_and_optimizer(model, optimizer, filename="model_checkpoint.pth"):
    """ Saves the model and optimizer state dictionaries to a file. """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, filename)


# Training function for policy gradients
def train_policy_gradient(policy, optimizer, states, actions, rewards):
    """
    Train the policy network using the policy gradient method adapted for both
    continuous and discrete actions.
    
    Args:
        policy (nn.Module): The policy network.
        optimizer (torch.optim.Optimizer): The optimizer.
        states (list of numpy.ndarray): The list of states.
        actions (list of tuple): A list of tuples, where each tuple contains continuous and discrete actions.
        rewards (list of float): The list of rewards.
    """
    
    # Calculate discounted rewards
    discounted_rewards = discount_rewards(rewards, gamma)
    
    # Convert everything to tensors
    states = torch.tensor(states, dtype=torch.float32)
    
    policy_losses = []
    for state, (continuous_actions, discrete_action), reward in zip(states, actions, discounted_rewards):
        state = state.unsqueeze(0)  # Add batch dimension
        means, std_devs, discrete_logits = policy(state)
        
        # Continuous actions
        continuous_distribution = Normal(means, std_devs)
        continuous_actions = torch.tensor(continuous_actions, dtype=torch.float32)
        continuous_log_probs = continuous_distribution.log_prob(continuous_actions).sum()

        # Discrete actions
        discrete_distribution = Categorical(logits=discrete_logits)
        discrete_log_prob = discrete_distribution.log_prob(torch.tensor([discrete_action], dtype=torch.int64))
        
        # Combine the log probabilities and multiply by the discounted reward
        total_log_prob = continuous_log_probs + discrete_log_prob
        loss = -total_log_prob * reward
        policy_losses.append(loss)
    
    # Perform backpropagation
    optimizer.zero_grad()
    total_loss = torch.stack(policy_losses).sum()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()  # Return the total loss for monitoring
    


    
    
def apply_actions_to_real_system(continuous_actions, discrete_actions):
    global max_time
    print(continuous_actions, discrete_actions)
    df = pd.read_excel("./2冷源流量补值代码-3种模式/补充混合+板换+冷机水流量.xlsx")
    new_max_time = df['MM-DD hour'].max()
    while True:  # This creates an infinite loop
        user_input = input("当操作执行完成，输入 'done': ")  # Prompt for input
        if user_input.lower() == 'done'and new_max_time != max_time:  # Check if the input is 'done' (case-insensitive)
            print("确认, 获取数据中...")
            break  # Exit the loop if the condition is met
        else:
            print("等待确认...")
            

        next_latest_time_rows = df[df['MM-DD hour'] == new_max_time]    

        next_state = []
        for system_id in range(1, num_systems + 1):
            if is_system_running(next_latest_time_rows,system_id):
                system_data = get_system_data(next_latest_time_rows,system_id)
            else:
                system_data = [0] * 23
                next_state.extend(system_data)

        next_state = next_state  # Apply normalization or standardization


        return next_state
    

    
    
def discount_rewards(rewards, gamma):
    discounted = []
    cumulative_total = 0
    # 从后向前计算折扣后的奖励总和
    for reward in rewards[::-1]:
        cumulative_total = reward + cumulative_total * gamma
        discounted.insert(0, cumulative_total)
    return discounted

#分割冷却塔功率
def process_and_sum_indices(current_state, indices):
        total_sum = 0
        for index in indices:
            if index < len(current_state):
                # Retrieve the value at the specified index
                value = current_state[index]

                # Check if the value is a string that needs to be split and summed
                if isinstance(value, str) and '/' in value:
                    # Split the string into parts based on '/'
                    parts = value.split('/')
                    # Convert each part to float and add to the total sum
                    total_sum += sum(float(part) for part in parts)
                else:
                    # For normal numeric values, add directly to the total sum
                    total_sum += float(value)
            else:
                print(f"Warning: Index {index} is out of range.")
    
        return total_sum

def is_chiller_on(state, index):
# Check if the chiller at the given index is on (temperature > 0)
    return state[index] > 0

#计算 reward
def calculate_reward_from_power_consumption(current_state,next_state):
    #R=w1×Consumption reduced+w2×Cooling Demand Met−w3×Deviation from Set Points−w4×Operational Extremes

    # List of indices we want to sum up, including index 18 with the complex string
    power_indices = [11, 13, 15, 20,21,22,23,40,42,44,49,50,51,52,69,71,73,78,79,80,81,98,100,102,107,108,109,110] 

    # Calculate the sum of the values at the specified indices
    total_old_power = process_and_sum_indices(current_state, power_indices)

    total_new_power = process_and_sum_indices(next_state,power_indices)

    # 奖励为功率减少的量，如果功率增加，则奖励为负
    consumption_reduced = total_old_power - total_new_power
    
    chiller_count_on_current = 0
    chiller_count_on_next = 0
    
    #6.7°C -13.3°C
    #R=w1×Efficiency+w2×Cooling Demand Met−w3×Temperature Deviation Penalty
    #Reward Efficiency: Assign a positive reward for higher outlet temperatures within the acceptable range, as these reduce the chiller load and energy consumption.
        
    # Indices corresponding to the outlet temperatures of different chillers/systems
    chiller_outlet_temp_indices = [1, 30, 59, 88]
    chiller_inlet_temp_indices=[0,29,58,87]
    chill_water_flow_indices=[24,53,82,111]
    chiller_power=[11,40,69,98]

    cool_efficiency_reward = 0
    COP_reward=0
    turn_off_reward=0
    for outlet_idx, inlet_idx, flow_idx,chill_power_idx in zip(chiller_outlet_temp_indices, chiller_inlet_temp_indices, chill_water_flow_indices,chiller_power):
        current_on = is_chiller_on(current_state, outlet_idx)
        next_on = is_chiller_on(next_state, outlet_idx)


        
        if current_on and not next_on:
            # Reward for turning off the chiller
            
            chiller_count_on_current +=1
        elif current_on and next_on:
            chiller_count_on_current +=1
            chiller_count_on_next += 1
            flow_rate = float(current_state[flow_idx])  
            T_in_current = float(current_state[inlet_idx])
            T_in_next=float(next_state[inlet_idx])
            T_out_current = float(current_state[outlet_idx])
            T_out_next = float(next_state[outlet_idx])
            c_p=4.186*1000/3600
            Q_actual_current = flow_rate * c_p * (T_in_current - T_out_current)
            Q_actual_next = flow_rate * c_p * (T_in_next - T_out_next)

            #turn_off reward
            if chiller_count_on_current>chiller_count_on_next:
                turn_off_reward+=chiller_count_on_current-chiller_count_on_next
            else:
                turn_off_reward-=chiller_count_on_next-chiller_count_on_current

            # Check if the temperature is higher and reward if it is
            if T_out_next > T_out_current:
                cool_efficiency_reward += T_out_next - T_out_current# temp_improvement_reward is a factor defined elsewhere
            if T_out_next < T_out_current:
                cool_efficiency_reward -= T_out_current-T_out_next
        # Optionally, handle cases where the chiller was off and remains off or turns on
        

        
            # Penalize if outside acceptable temperature range
            if float(next_state[outlet_idx])<6.7:
                Operational_Extremes=-(6.7-float(next_state[outlet_idx])) # penalty for overcooling
            if float(next_state[outlet_idx])>13.3:
                Operational_Extremes=-(float(next_state[outlet_idx])-13.3)  # penalty for undercooling
            
        
            COP_current= Q_actual_current / float(current_state[chill_power_idx])
            COP_next=Q_actual_next/float(next_state[chill_power_idx]) 

            if COP_next > COP_current:
                COP_reward +=COP_next-COP_current# temp_improvement_reward is a factor defined elsewhere
            if COP_next < COP_current:
                COP_reward -= COP_current-COP_next
            
    
    
    
    reward=consumption_reduced+turn_off_reward+cool_efficiency_reward+ COP_reward - Operational_Extremes
    #还差设定值的偏移
    return reward

    

# Training loop
def main_training_loop():
    step=1
    while True:
        # Retrieve the current state of the real environment
        current_state,current_cooling_demand = get_real_system_state()  # This function needs to be implemented to fetch real-world data
        # Initialize episode memory
        states, actions, rewards = [], [], []
        
        
        state_tensor = torch.from_numpy(normalize(current_state)).float().unsqueeze(0)
        means, std_devs, discrete_logits = policy(state_tensor)

        # Sample from the distributions for continuous actions
        continuous_distribution = Normal(means, std_devs)
        continuous_actions = continuous_distribution.sample()
        continuous_actions=scale_continuous_actions(continuous_actions)
        # Apply constraints to continuous actions here if necessary
        # ...

        # Sample from the distribution for discrete actions
        discrete_distribution = Categorical(logits=discrete_logits)
        discrete_action_index = discrete_distribution.sample()
        discrete_action = apply_discrete_action(discrete_action_index)
        # Combine and possibly constrain the discrete actions here
        # ...

        # Apply the actions to the real system and get the new state and reward
        # Ensure this is done in a safe manner with proper error checking
        next_state= apply_actions_to_real_system(continuous_actions, discrete_action)
        
        # ...

        states.append(current_state)
        actions.append((continuous_actions, discrete_action))  # Store actions taken

#         Calculate reward based on the power consumption difference
        reward = calculate_reward_from_power_consumption(current_state,next_state)
#         ...

        # Update policy after each episode or after collecting enough data
        train_policy_gradient(policy, optimizer, states, actions, rewards)
        save_model_and_optimizer(policy, optimizer, filename=f"/Users/zhiranbai/Documents/GitHub/Chiller-plate-optimizatoin/model_checkpoint.pth")
        rewards.append(reward)
        current_state = normalize(next_state)
        step+=1

main_training_loop()