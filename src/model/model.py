import torch
import torch.nn as nn
import snntorch as snn


# ============================
# Network Input Layer with Synaptic input current
# ============================
class InputLayer(nn.Module):
    def __init__(self, 
                 neurons,
                 alpha_syn, 
                 beta_syn, 
                 beta, 
                 threshold_in=1.0, 
                 reset_in='zero', 
                 refractory_period_in=0, 
                 device=None
                 ):
        super().__init__()

        # set device if not set from external
        self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                            "cuda" if torch.cuda.is_available() else 
                            "cpu") if device is None else device

        # Input Synapses
        self.input_neurons = neurons
        self.alpha_syn = alpha_syn 
        self.beta_syn = beta_syn
        
        # Neurons 
        self.beta = beta
        self.threshold_in = threshold_in
        self.refractory_period_in = refractory_period_in
        self.reset_in = reset_in
        
        
        # Refractory timers (each neuron has its own timer)
        self.refractory_timer_in = torch.zeros(neurons, device=self.device)

        # Synaptic + LIF
        self.synapse = snn.Synaptic(alpha=alpha_syn, beta=beta_syn).to(self.device)
        self.lif = snn.Leaky(beta=beta, threshold=threshold_in, reset_mechanism=reset_in).to(self.device)


    def forward(self, spikes):
        
        # Compute synaptic currents
        I_syn = self.synapse(spikes)[0]
        
        self.refractory_timer_in = torch.clamp(self.refractory_timer_in - 1, min=0)

        mask_in = (self.refractory_timer_in == 0)

        # LIF neurons (apply refractory mask)
        spikes_out, mem_out = self.lif(I_syn)

        spikes_out = spikes_out * mask_in
        
        self.refractory_timer_in = torch.where(spikes_out > 0,
            torch.tensor(self.refractory_timer_in).detach().to(self.device),
            self.refractory_timer_in
        )
        
        return mem_out, spikes_out, I_syn, spikes
    
    def info(self):
        return {
                    "synapse": {
                        "neurons": self.input_neurons,
                        "alpha": self.alpha_syn,
                        "beta": self.beta_syn
                    },
                    "lif": {
                        "neurons": self.input_neurons,
                        "beta": self.beta,
                        "threshold": self.threshold_in,
                        "reset_mechanism": self.reset_in
                    }
                }
        
        
        
        

# ============================
# Network Hidden Layer
# ============================
class HiddenLayer(nn.Module):
    def __init__(self, 
                 input_neurons,
                 output_neurons,
                 beta,
                 threshold_out=0.5,
                 threshold_min=0.5, 
                 threshold_max=1.0,
                 threshold_inc=0.001, 
                 threshold_dec=0,
                 adaptive_threshold_on=True,
                 reset_out='zero',
                 inhibition=False, 
                 inhibition_strength=0.5,
                 refractory_period=0,
                 initial_weights=None, 
                 random_weights_range=[0.3, 0.5],
                 device=None
                 ):
        super().__init__()

        # set device if not set from external
        self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                            "cuda" if torch.cuda.is_available() else 
                            "cpu") if device is None else device

        self.input_neurons = input_neurons
        
        # Output Neurons
        self.output_neurons = output_neurons
        self.beta = beta
        self.threshold_out = threshold_out
        self.refractory_period = refractory_period
        self.reset_out = reset_out
        self.inhibition = inhibition
        
        # Adaptive Threshold
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.threshold_dec = threshold_dec
        self.threshold_inc = threshold_inc
        self.adaptive_threshold_on = adaptive_threshold_on
        
        # Refractory timers (each neuron has its own timer)
        self.refractory_timer_out = torch.zeros(output_neurons, device=self.device)

        # input-to-output weights
        if initial_weights is None:
            self.weights = nn.Parameter(
                torch.rand(output_neurons, input_neurons, device=self.device) * (random_weights_range[1] - random_weights_range[0]) + random_weights_range[0]
                )
        else:
            self.weights = nn.Parameter(initial_weights)

        # Output neurons
        self.lif_out = snn.Leaky(beta=beta, threshold=threshold_out, reset_mechanism=reset_out, inhibition=inhibition).to(self.device)
        
        # Adaptive threshold
        self.adaptive_threshold = threshold_out * torch.ones(output_neurons, device=self.device)
        self.register_buffer('lif_threshold_buffer', self.adaptive_threshold.clone().detach())

        # Inhibition strength
        self.inhibition_strength = inhibition_strength 

    def forward(self, spikes):
        
        # Projection to Output Layer
        Iw_in = torch.matmul(spikes, self.weights.T)
        
        # **Aggiorna dinamicamente la soglia di Leaky**
        #self.lif_out.register_buffer('threshold', self.adaptive_threshold.clone().detach())
        self.lif_threshold_buffer.copy_(self.adaptive_threshold)
        self.lif_out.threshold = self.lif_threshold_buffer

        # Output Layer
        spk_out, mem_out = self.lif_out(Iw_in)

        # **Lateral Inhibition**
        if spk_out.any() and self.inhibition_strength > 0:
            mask = (spk_out == 0)  
            inhibition = spk_out * self.inhibition_strength  
            mem_out -= mask * inhibition.sum(dim=-1, keepdim=True)  

        # **Dynamic Threshold Update**
        if self.adaptive_threshold_on:
            with torch.no_grad():
                self.adaptive_threshold -= self.threshold_dec
                self.adaptive_threshold += spk_out.squeeze(0) * self.threshold_inc  
                self.adaptive_threshold = torch.clamp(self.adaptive_threshold, min=self.threshold_min, max=self.threshold_max)

        # **Refractory Mechanism for Output Layer**
        self.refractory_timer_out = torch.clamp(self.refractory_timer_out - 1, min=0)
        mask_out = (self.refractory_timer_out == 0)
        spk_out = spk_out * mask_out

        self.refractory_timer_out = torch.where(spk_out > 0,
            torch.tensor(self.refractory_period, device=self.device),
            self.refractory_timer_out
        )

        return mem_out, spk_out, Iw_in, spikes, self.adaptive_threshold
    
    def info(self):
        return {
                    "neurons": self.output_neurons,
                    "beta": self.beta,
                    "threshold_base": self.threshold_out,
                    "adaptive_threshold_increase": self.threshold_inc,
                    "adaptive_threshold_max": self.threshold_max,
                    "adaptive_threshold_on": self.adaptive_threshold_on,
                    "reset_mechanism": self.reset_out,
                    "latheral_inhibition": self.inhibition,
                    "inhibition_strength": self.inhibition_strength
                }
        
        
        
# ============================
# Network Output Layer
# ============================
class OutputLayer(nn.Module):
    def __init__(self, 
                 input_neurons,
                 output_neurons,
                 reward_neurons,
                 beta_out,
                 threshold_out=0.5,
                 threshold_min=0.5, 
                 threshold_max=1.0,
                 threshold_inc=0.001, 
                 threshold_dec=0,
                 adaptive_threshold_on=True,
                 reset_out='zero',
                 reward_strength=0.5,
                 inhibition=False, 
                 inhibition_strength=0.5,
                 refractory_period_out=0,
                 initial_weights=None, 
                 random_weights_range=[0.3, 0.5],
                 device=None
                 ):
        super().__init__()

        # set device if not set from external
        self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                            "cuda" if torch.cuda.is_available() else 
                            "cpu") if device is None else device
        
        # Input Neurons
        self.input_neurons = input_neurons
        
        # Reward Neurons
        self.reward_neurons = reward_neurons
        self.reward_strength = reward_strength
        
        # Output Neurons
        self.output_neurons = output_neurons
        self.beta_out = beta_out
        self.threshold_out = threshold_out
        self.refractory_period_out = refractory_period_out
        self.reset_out = reset_out
        self.inhibition = inhibition
        
        # Adaptive Threshold
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.threshold_dec = threshold_dec
        self.threshold_inc = threshold_inc
        self.adaptive_threshold_on = adaptive_threshold_on
        
        # Refractory timers (each neuron has its own timer)
        self.refractory_timer_out = torch.zeros(output_neurons, device=self.device)
        

        # input-to-output weights
        if initial_weights is None:
            self.weights = nn.Parameter(
                torch.rand(output_neurons, input_neurons, device=self.device) * (random_weights_range[1] - random_weights_range[0]) + random_weights_range[0]
                )
        else:
            self.weights = nn.Parameter(initial_weights)
            
            
        # reward-to-output weights
        self.reward_weights = nn.Parameter(torch.ones(output_neurons, reward_neurons, device=self.device) * self.reward_strength)

        # Output neurons
        self.lif_out = snn.Leaky(beta=beta_out, threshold=threshold_out, reset_mechanism=reset_out, inhibition=inhibition).to(self.device)
        #self.lif_out = snn.Alpha(beta=beta_out, alpha=0.6, threshold=threshold_out, reset_mechanism=reset_out, inhibition=inhibition).to(self.device)

        # Adaptive threshold
        self.adaptive_threshold = threshold_out * torch.ones(output_neurons, device=self.device)

        # Inhibition strength
        self.inhibition_strength = inhibition_strength 

    def forward(self, spikes):
        
        # Projection to Output Layer
        Iw_in = torch.matmul(spikes, self.weights.T)
        
        # **Aggiorna dinamicamente la soglia di Leaky**
        self.lif_out.register_buffer('threshold', self.adaptive_threshold.clone().detach())
        
        # Output Layer
        spk_out, mem_out = self.lif_out(Iw_in)
        
        
        # **Lateral Inhibition**
        if spk_out.any() and self.inhibition_strength > 0:
            mask = (spk_out == 0)  
            inhibition = spk_out * self.inhibition_strength  
            mem_out -= mask * inhibition.sum(dim=-1, keepdim=True)  

        # **Dynamic Threshold Update**
        if self.adaptive_threshold_on:
            with torch.no_grad():
                self.adaptive_threshold -= self.threshold_dec
                self.adaptive_threshold += spk_out.squeeze(0) * self.threshold_inc  
                self.adaptive_threshold = torch.clamp(self.adaptive_threshold, min=self.threshold_min, max=self.threshold_max)

        # **Refractory Mechanism for Output Layer**
        self.refractory_timer_out = torch.clamp(self.refractory_timer_out - 1, min=0)
        mask_out = (self.refractory_timer_out == 0)
        spk_out = spk_out * mask_out

        self.refractory_timer_out = torch.where(spk_out > 0,
            torch.tensor(self.refractory_period_out, device=self.device),
            self.refractory_timer_out
        )
        
        
        return mem_out, spk_out, spikes, Iw_in, self.adaptive_threshold

    
    def info(self):
        return {
                "output": {
                    "neurons": self.output_neurons,
                    "beta": self.beta_out,
                    "threshold_base": self.threshold_out,
                    "adaptive_threshold_increase": self.threshold_inc,
                    "adaptive_threshold_max": self.threshold_max,
                    "reset_mechanism": self.reset_out,
                    "reward_strength": self.reward_strength,
                    "latheral_inhibition": self.inhibition,
                    "inhibition_strength": self.inhibition_strength
                }
        }
        
        
# ============================
# STDP Rule with Two Pre-Synaptic Traces, A_plus / A_minus, and Spike Shift
# ============================
class STDP_prev:
    def __init__(self, layer, lr=0.01, tau_pre1=5.0, tau_pre2=100.0, tau_post=5.0, A_plus=0.01, A_minus=-0.005, shift=0):
        self.lr = lr
        self.tau_pre1 = tau_pre1  # Traccia a breve termine
        self.tau_pre2 = tau_pre2  # Traccia a lungo termine
        self.tau_post = tau_post
        self.A_plus = A_plus  # Potenziamento sinaptico (LTP)
        self.A_minus = A_minus  # Depressione sinaptica (LTD)
        self.shift = shift  # Offset temporale per STDP
        self.layer = layer
        
        # Tracce pre e post-sinaptiche
        self.pre_trace1 = torch.zeros(layer.input_neurons, device=layer.device)
        self.pre_trace2 = torch.zeros(layer.input_neurons, device=layer.device)
        self.post_trace = torch.zeros(layer.output_neurons, device=layer.device)
        
        # Buffer per shift temporale degli spike pre-sinaptici
        self.pre_spike_buffer = []

    def update(self, spikes_pre, spikes_post):
        # Evita divisione per zero nei tau
        tau_pre1_safe = max(self.tau_pre1, 1e-6)
        tau_pre2_safe = max(self.tau_pre2, 1e-6)
        tau_post_safe = max(self.tau_post, 1e-6)
        
        # Salva lo spike corrente nel buffer
        if self.shift > 0:
            self.pre_spike_buffer.append(spikes_pre.clone())
            if len(self.pre_spike_buffer) > self.shift:
                spikes_pre = self.pre_spike_buffer.pop(0)  # Usa lo spike ritardato
            else:
                return  # Aspetta fino a quando il buffer è pieno
        
        # Aggiornamento delle tracce pre-sinaptiche (breve e lunga)
        self.pre_trace1 = self.pre_trace1 * torch.exp(torch.tensor(-1.0 / tau_pre1_safe, device=self.layer.device)) + spikes_pre.mean(dim=0)
        self.pre_trace2 = self.pre_trace2 * torch.exp(torch.tensor(-1.0 / tau_pre2_safe, device=self.layer.device)) + spikes_pre.mean(dim=0)
        
        # Aggiornamento della traccia post-sinaptica
        self.post_trace = self.post_trace * torch.exp(torch.tensor(-1.0 / tau_post_safe, device=self.layer.device)) + spikes_post.mean(dim=0)
        
        # Calcolo del cambiamento di peso usando A_plus e A_minus
        dW_in = self.lr * (self.A_plus * torch.outer(spikes_post.mean(dim=0), self.pre_trace1) - self.A_minus * torch.outer(self.post_trace, self.pre_trace2))
        
        # Aggiornamento dei pesi del modello
        self.layer.weights.data += dW_in
        self.layer.weights.data = torch.clamp(self.layer.weights, 0, 1)
        
    def info(self):
        return {
            "stdp": {
                    "lr": self.lr,
                    "tau_pre1": self.tau_pre1,
                    "tau_pre2": self.tau_pre2,
                    "tau_post": self.tau_post,
                    "A_plus": self.A_plus,
                    "A_minus": self.A_minus
                }
        }
        
        
# ============================
# STDP with Two Pre-Synaptic Traces + Optional Passive LTD for WTA
# ============================
class STDP:
    def __init__(self, layer, lr=0.01, tau_pre1=5.0, tau_pre2=100.0, tau_post=5.0,
                 A_plus=0.01, A_minus=-0.005, shift=0, A_passive=0.02, use_passive_ltd=True):
        self.lr = lr
        self.tau_pre1 = tau_pre1  # Short-term pre-trace
        self.tau_pre2 = tau_pre2  # Long-term pre-trace
        self.tau_post = tau_post
        self.A_plus = A_plus  # Potentiation
        self.A_minus = A_minus  # Depression
        self.A_passive = A_passive  # Passive LTD for inactive pre
        self.shift = shift
        self.use_passive_ltd = use_passive_ltd
        self.layer = layer

        # Traces
        self.pre_trace1 = torch.zeros(layer.input_neurons, device=layer.device)
        self.pre_trace2 = torch.zeros(layer.input_neurons, device=layer.device)
        self.post_trace = torch.zeros(layer.output_neurons, device=layer.device)

        # Spike buffer for pre shift
        self.pre_spike_buffer = []

    def update(self, spikes_pre, spikes_post):
        tau_pre1_safe = max(self.tau_pre1, 1e-6)
        tau_pre2_safe = max(self.tau_pre2, 1e-6)
        tau_post_safe = max(self.tau_post, 1e-6)

        # Apply temporal shift if needed
        if self.shift > 0:
            self.pre_spike_buffer.append(spikes_pre.clone())
            if len(self.pre_spike_buffer) > self.shift:
                spikes_pre = self.pre_spike_buffer.pop(0)
            else:
                return  # wait for buffer

        # Mean over batch
        spikes_pre_mean = spikes_pre.mean(dim=0)
        spikes_post_mean = spikes_post.mean(dim=0)

        # Update traces
        self.pre_trace1 = self.pre_trace1 * torch.exp(torch.tensor(-1.0 / tau_pre1_safe, device=self.layer.device)) + spikes_pre_mean
        self.pre_trace2 = self.pre_trace2 * torch.exp(torch.tensor(-1.0 / tau_pre2_safe, device=self.layer.device)) + spikes_pre_mean
        self.post_trace = self.post_trace * torch.exp(torch.tensor(-1.0 / tau_post_safe, device=self.layer.device)) + spikes_post_mean

        # STDP weight update
        ltp = self.A_plus * torch.outer(spikes_post_mean, self.pre_trace1)
        ltd = self.A_minus * torch.outer(self.post_trace, self.pre_trace2)
        dW = self.lr * (ltp - ltd)

        # Optional: WTA-style passive LTD
        if self.use_passive_ltd and self.A_passive != 0:
            inactive_pre = 1.0 - spikes_pre_mean  # shape: [input]
            active_post = spikes_post_mean        # shape: [output]
            passive_ltd = torch.outer(active_post, inactive_pre)
            dW -= self.lr * self.A_passive * passive_ltd

        # Update weights
        self.layer.weights.data += dW
        self.layer.weights.data = torch.clamp(self.layer.weights.data, 0.0, 1.0)

    def info(self):
        return {
            "stdp": {
                "lr": self.lr,
                "tau_pre1": self.tau_pre1,
                "tau_pre2": self.tau_pre2,
                "tau_post": self.tau_post,
                "A_plus": self.A_plus,
                "A_minus": self.A_minus,
                "A_passive": self.A_passive,
                "use_passive_ltd": self.use_passive_ltd,
                "shift": self.shift
            }
        }
        
        
        
        
# ============================
# R-STDP with Eligibility Trace and Delayed Reward - Inverted STDP + Reward-Modulated STDP (R-STDP)
# ============================
class STDP_ET_prev:
    def __init__(self, layer, lr=0.01, tau_pre1=5.0, tau_pre2=100.0, tau_post=5.0, tau_e=1000.0,
                 A_plus=0.01, A_minus=0.005, shift=0, passive_ltd=None):
        self.lr = lr
        self.tau_pre1 = tau_pre1
        self.tau_pre2 = tau_pre2
        self.tau_post = tau_post
        self.tau_e = tau_e
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.shift = shift
        self.passive_ltd = passive_ltd  # Se None, non viene usata
        self.layer = layer

        self.pre_trace1 = torch.zeros(layer.input_neurons, device=layer.device)
        self.pre_trace2 = torch.zeros(layer.input_neurons, device=layer.device)
        self.post_trace = torch.zeros(layer.output_neurons, device=layer.device)
        self.eligibility = torch.zeros(layer.output_neurons, layer.input_neurons, device=layer.device)
        self.pre_spike_buffer = []

    def update(self, spikes_pre, spikes_post):
        # Sicurezza sui tau
        tau_pre1_safe = max(self.tau_pre1, 1e-6)
        tau_pre2_safe = max(self.tau_pre2, 1e-6)
        tau_post_safe = max(self.tau_post, 1e-6)
        tau_e_safe = max(self.tau_e, 1e-6)

        # Shift temporale, se necessario
        if self.shift > 0:
            self.pre_spike_buffer.append(spikes_pre.clone())
            if len(self.pre_spike_buffer) > self.shift:
                spikes_pre = self.pre_spike_buffer.pop(0)
            else:
                return

        # Decadimento delle tracce
        self.pre_trace1 *= torch.exp(torch.tensor(-1.0 / tau_pre1_safe, device=self.layer.device))
        self.pre_trace2 *= torch.exp(torch.tensor(-1.0 / tau_pre2_safe, device=self.layer.device))
        self.post_trace *= torch.exp(torch.tensor(-1.0 / tau_post_safe, device=self.layer.device))
        self.eligibility *= torch.exp(torch.tensor(-1.0 / tau_e_safe, device=self.layer.device))

        # Aggiorna le tracce
        self.pre_trace1 += spikes_pre.mean(dim=0)
        self.pre_trace2 += spikes_pre.mean(dim=0)
        self.post_trace += spikes_post.mean(dim=0)

        # === LTD: applicata subito ai pesi ===
        if self.A_minus != 0:
            ltd = self.A_minus * torch.outer(self.post_trace, self.pre_trace2)
            self.layer.weights.data -= ltd
            self.layer.weights.data = torch.clamp(self.layer.weights.data, 0.05, 1.0)

        # === LTP: accumulata nella eligibility trace (solo potenziale) ===
        if self.A_plus != 0:
            ltp = self.A_plus * torch.outer(spikes_post.mean(dim=0), self.pre_trace1)
            self.eligibility += ltp

    def apply_reward(self, reward):
        if isinstance(reward, (float, int)):
            reward = torch.full((self.layer.neurons, 1), reward, device=self.layer.device)
        elif isinstance(reward, torch.Tensor) and reward.ndim == 1:
            reward = reward.view(-1, 1)

        self.layer.weights.data += self.lr * reward * self.eligibility
        self.layer.weights.data = torch.clamp(self.layer.weights.data, 0.0, 1.0)

    def info(self):
        return {
            "rstdp": {
                "lr": self.lr,
                "tau_pre1": self.tau_pre1,
                "tau_pre2": self.tau_pre2,
                "tau_post": self.tau_post,
                "tau_e": self.tau_e,
                "A_plus": self.A_plus,
                "A_minus": self.A_minus,
                "shift": self.shift,
                "passive_ltd": self.passive_ltd
            }
        }
        
 
# ============================
# R-STDP with Eligibility Trace and Delayed Reward - Inverted STDP + Reward-Modulated STDP (R-STDP)
# ============================        
class STDP_ET:
    def __init__(self, layer, lr=0.01, tau_pre1=5.0, tau_pre2=100.0, tau_post=5.0, tau_e=1000.0,
                 A_plus=0.01, A_minus=0.005, shift=0, passive_ltd=None, classic_stdp_only=False):
        self.lr = lr
        self.tau_pre1 = tau_pre1
        self.tau_pre2 = tau_pre2
        self.tau_post = tau_post
        self.tau_e = tau_e
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.shift = shift
        self.passive_ltd = passive_ltd
        self.classic_stdp_only = classic_stdp_only
        self.layer = layer

        self.pre_trace1 = torch.zeros(layer.input_neurons, device=layer.device)
        self.pre_trace2 = torch.zeros(layer.input_neurons, device=layer.device)
        self.post_trace = torch.zeros(layer.output_neurons, device=layer.device)
        self.eligibility = torch.zeros(layer.output_neurons, layer.input_neurons, device=layer.device)
        self.pre_spike_buffer = []

    def update(self, spikes_pre, spikes_post):
        tau_pre1_safe = max(self.tau_pre1, 1e-6)
        tau_pre2_safe = max(self.tau_pre2, 1e-6)
        tau_post_safe = max(self.tau_post, 1e-6)
        tau_e_safe = max(self.tau_e, 1e-6)

        # Shift temporale (ritardo pre-sinaptico)
        if self.shift > 0:
            self.pre_spike_buffer.append(spikes_pre.clone())
            if len(self.pre_spike_buffer) > self.shift:
                spikes_pre = self.pre_spike_buffer.pop(0)
            else:
                return

        # Decadimento delle tracce
        self.pre_trace1 *= torch.exp(torch.tensor(-1.0 / tau_pre1_safe, device=self.layer.device))
        self.pre_trace2 *= torch.exp(torch.tensor(-1.0 / tau_pre2_safe, device=self.layer.device))
        self.post_trace *= torch.exp(torch.tensor(-1.0 / tau_post_safe, device=self.layer.device))
        if not self.classic_stdp_only:
            self.eligibility *= torch.exp(torch.tensor(-1.0 / tau_e_safe, device=self.layer.device))

        # Aggiorna tracce
        self.pre_trace1 += spikes_pre.mean(dim=0)
        self.pre_trace2 += spikes_pre.mean(dim=0)
        self.post_trace += spikes_post.mean(dim=0)

        # LTD
        if self.A_minus != 0:
            ltd = self.A_minus * torch.outer(self.post_trace, self.pre_trace2)
            self.layer.weights.data -= ltd
            self.layer.weights.data = torch.clamp(self.layer.weights.data, 0.05, 1.0)

        # LTP
        if self.A_plus != 0:
            ltp = self.A_plus * torch.outer(spikes_post.mean(dim=0), self.pre_trace1)
            if self.classic_stdp_only:
                # Applica subito ltp ai pesi
                self.layer.weights.data += self.lr * ltp
                self.layer.weights.data = torch.clamp(self.layer.weights.data, 0.0, 1.0)
            else:
                # Accumula in eligibility trace
                self.eligibility += ltp

    def apply_reward(self, reward):
        if self.classic_stdp_only:
            return  # Nessun reward in modalità STDP classico

        if isinstance(reward, (float, int)):
            reward = torch.full((self.layer.output_neurons, 1), reward, device=self.layer.device)
        elif isinstance(reward, torch.Tensor) and reward.ndim == 1:
            reward = reward.view(-1, 1)

        self.layer.weights.data += self.lr * reward * self.eligibility
        self.layer.weights.data = torch.clamp(self.layer.weights.data, 0.0, 1.0)

    def info(self):
        return {
            "rstdp": {
                "lr": self.lr,
                "tau_pre1": self.tau_pre1,
                "tau_pre2": self.tau_pre2,
                "tau_post": self.tau_post,
                "tau_e": self.tau_e,
                "A_plus": self.A_plus,
                "A_minus": self.A_minus,
                "shift": self.shift,
                "passive_ltd": self.passive_ltd,
                "classic_stdp_only": self.classic_stdp_only
            }
        }
        
        
        

# ============================
# RewardBasedModel con gestione memoria migliorata
# ============================

class RewardBasedModel(nn.Module):
    def __init__(self, 
                 input_neurons=64, 
                 hidden_neurons=400, 
                 output_neurons=5, 
                 reward_neurons=1, 
                 lr_un=0.01, # was 0.001
                 dt=1.0,
                 device=None):
        super().__init__()

        self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                            "cuda" if torch.cuda.is_available() else 
                            "cpu") if device is None else device

        self.input_layer = InputLayer(input_neurons,
                                      0.99 ** dt, 
                                      0.85 ** dt, 
                                      0.5 ** dt,
                                      refractory_period_in=15,
                                      device=device).to(device)

        self.hidden_layer = HiddenLayer(input_neurons,
                                        hidden_neurons,
                                        0.8 ** dt,
                                        threshold_max = 20,
                                        adaptive_threshold_on=True,
                                        inhibition=True, 
                                        inhibition_strength=0.8,
                                        refractory_period=5,
                                        random_weights_range=[0.3, 0.5], # [0.3, 0.5]
                                        device=device).to(device)

        self.output_layer = OutputLayer(hidden_neurons,
                                        output_neurons,
                                        reward_neurons,
                                        0.5 ** dt, # 0.7
                                        threshold_out=0.5,
                                        threshold_min=0.1, 
                                        threshold_max=20.0,
                                        threshold_inc=0.001, 
                                        threshold_dec=0,
                                        adaptive_threshold_on=False,
                                        reward_strength=1.0,
                                        inhibition=True, 
                                        inhibition_strength=1.5,
                                        refractory_period_out=5, # 5
                                        random_weights_range=[0.3, 0.5], # [0.3, 0.5]
                                        device=device).to(device)

        self.stdp = STDP(self.hidden_layer,
                         lr=lr_un,
                         tau_pre1=5.0,
                         tau_pre2=40.0,
                         tau_post=5.0,
                         A_plus=0.01,
                         A_minus=-0.005)

        self.shift_stdp = STDP_ET(self.output_layer,
                                  lr=0.1,
                                  tau_pre1=10.0,
                                  tau_pre2=100.0,
                                  tau_post=10.0, 
                                  A_plus=0.2,
                                  A_minus=0.00001,
                                  passive_ltd=None,
                                  tau_e=1000.0
                                )

        self.configuration_type = None
        self.enable_hidden_learning = True
        self.enable_output_layer = True
        self.enable_output_learning = True
        self.enable_reward = True

    def train_unsupervised(self):
        self.configuration_type = 'train_unsupervised'
        self.enable_hidden_learning = True
        self.hidden_layer.adaptive_threshold_on = True
        self.output_layer.adaptive_threshold_on = False
        self.enable_output_layer = True
        self.enable_output_learning = False
        self.enable_reward = False
        self.shift_stdp.classic_stdp_only = False
        
    def train_unsupervised_2_phases(self):
        self.configuration_type = 'train_unsupervised_2_phases'
        self.enable_hidden_learning = True
        self.hidden_layer.adaptive_threshold_on = True
        self.output_layer.adaptive_threshold_on = True
        self.enable_output_layer = False
        self.enable_output_learning = False
        self.enable_reward = False
        self.shift_stdp.classic_stdp_only = True
        
    def train_unsupervised_3_phases(self):
        self.configuration_type = 'train_unsupervised_3'
        self.enable_hidden_learning = True
        self.hidden_layer.adaptive_threshold_on = True
        self.output_layer.adaptive_threshold_on = True
        self.enable_output_layer = True
        self.enable_output_learning = True
        self.enable_reward = True
        self.shift_stdp.classic_stdp_only = True

    def test_unsupervised(self):
        self.configuration_type = 'test_unsupervised'
        self.enable_hidden_learning = False
        self.hidden_layer.adaptive_threshold_on = False
        self.output_layer.adaptive_threshold_on = False
        self.enable_output_layer = False
        self.enable_output_learning = False
        self.enable_reward = False
        self.shift_stdp.classic_stdp_only = False
        
    def finetune_unsupervised(self):
        self.configuration_type = 'finetune_unsupervised'
        self.enable_hidden_learning = False
        self.hidden_layer.adaptive_threshold_on = False
        self.output_layer.adaptive_threshold_on = False
        self.enable_output_layer = True
        self.enable_output_learning = False
        self.enable_reward = False
        self.shift_stdp.classic_stdp_only = False

    def train_reward(self):
        self.configuration_type = 'train_reward'
        self.hidden_layer.adaptive_threshold_on = False
        self.output_layer.adaptive_threshold_on = False
        self.enable_hidden_learning = False
        self.enable_output_layer = True
        self.enable_output_learning = True
        self.enable_reward = True
        self.shift_stdp.classic_stdp_only = False

    def test_reward(self):
        self.configuration_type = 'test_reward'
        self.hidden_layer.adaptive_threshold_on = False
        self.output_layer.adaptive_threshold_on = False
        self.enable_hidden_learning = False
        self.enable_output_layer = True
        self.enable_output_learning = False
        self.enable_reward = True
        self.shift_stdp.classic_stdp_only = False
    

    def online(self):
        self.configuration_type = 'online'
        self.hidden_layer.adaptive_threshold_on = False
        self.output_layer.adaptive_threshold_on = False
        self.enable_hidden_learning = False
        self.enable_output_layer = True
        self.enable_output_learning = True
        self.enable_reward = True
        self.shift_stdp.classic_stdp_only = False

    def forward(self, spikes, reward=None):
        input_layer_return = None
        hidden_layer_return = None
        output_layer_return = None

        with torch.no_grad():
            
            if spikes.shape[0] > 1:
                spikes = spikes.sum(dim=0, keepdim=True)
            
            input_layer_return = self.input_layer(spikes)
            spikes_input = input_layer_return[1].detach()

            hidden_layer_return = self.hidden_layer(spikes_input)
            spikes_hidden = hidden_layer_return[1].detach()

        if self.enable_hidden_learning:
            self.stdp.update(spikes_input, spikes_hidden)

        if self.enable_output_layer:
            with torch.no_grad():
                output_layer_return = self.output_layer(spikes_hidden)
                spikes_output = output_layer_return[1].detach()

            if self.enable_output_learning:
                self.shift_stdp.update(spikes_hidden, spikes_output)

                if self.enable_reward and reward is not None:
                    self.shift_stdp.apply_reward(reward)

        return input_layer_return, hidden_layer_return, output_layer_return




        
        
        
    
    