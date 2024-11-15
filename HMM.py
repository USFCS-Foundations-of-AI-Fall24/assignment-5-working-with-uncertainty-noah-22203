import random
import sys
import argparse

# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# HMM model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""



        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        self.transitions = {}
        self.emissions = {}

        with open(f"{basename}.trans", "r") as f:
            for line in f:
                parts = line.strip().split()
                state = parts[0]
                if state not in self.transitions:
                    self.transitions[state] = {}
                for i in range(1, len(parts), 2):
                    next_state = parts[i]
                    prob = parts[i + 1]
                    self.transitions[state][next_state] = prob

        
        with open(f"{basename}.emit", "r") as f:
            for line in f:
                parts = line.strip().split()
                state = parts[0]
                if state not in self.emissions:
                    self.emissions[state] = {}
                for i in range(1, len(parts), 2):
                    output = parts[i]
                    prob = parts[i + 1]
                    self.emissions[state][output] = prob


    def generate(self, n):
        sequence1 = []
        sequence2 = []
        state = '#'
        for _ in range(n):
            next_states = list(self.transitions[state].keys())
          
            next_probs = [float(prob) for prob in self.transitions[state].values()]  
            state = random.choices(next_states, weights=next_probs, k=1)[0]

            emissions = list(self.emissions[state].keys())
            emission_probs = [float(prob) for prob in self.emissions[state].values()]  
            emission = random.choices(emissions, weights=emission_probs, k=1)[0]

            sequence1.append(state)
            sequence2.append(emission)
        sequence = Sequence(sequence1, sequence2)
        return sequence


    def forward(self, sequence):
        observations = sequence.outputseq
        states = list(self.transitions.keys())
        fwd = [[0 for _ in range(len(observations) + 1)] for _ in range(len(states))]
        fwd[0][0] = 1
        
        if '#' in states: 
            states.remove('#')
            states.insert(0, '#')

        for i in range(1, len(states)):
            fwd[i][1] = float(self.transitions['#'].get(states[i], 0)) * float(self.emissions[states[i]].get(observations[0], 0))

        for i in range(2, len(observations) + 1):
            for j in range(1, len(states)):
                sum = 0
                for k in range(1, len(states)):
                    sum += fwd[k][i - 1] * float(self.transitions[states[k]].get(states[j], 0)) * float(self.emissions[states[j]].get(observations[i - 1], 0))
                fwd[j][i] = sum
        
        max_index = 0
        max_prob = 0

        for i in range(1, len(states)):
            if fwd[i][len(observations)] > max_prob:
                max_prob = fwd[i][len(observations)]
                max_index = i

        return states[max_index]


    def viterbi(self, sequence):
        observations = sequence.outputseq
        states = list(self.transitions.keys())
        fwd = [[0 for _ in range(len(observations) + 1)] for _ in range(len(states))]
        fwd[0][0] = 1
        backpointers = [[0 for _ in range(len(observations) + 1)] for _ in range(len(states))]

        if '#' in states: 
            states.remove('#')
            states.insert(0, '#')
            
        for i in range(1, len(states)):
            fwd[i][1] = float(self.transitions['#'].get(states[i], 0)) * float(self.emissions[states[i]].get(observations[0], 0))

        for i in range(2, len(observations) + 1):
            for j in range(1, len(states)):
                max = 0
                max_index = 0
                for k in range(1, len(states)):
                    temp = fwd[k][i - 1] * float(self.transitions[states[k]].get(states[j], 0)) * float(self.emissions[states[j]].get(observations[i - 1], 0))
                    if temp > max:
                        max = temp
                        max_index = k
                fwd[j][i] = max
                backpointers[j][i] = max_index
        
        max_index = 0
        max_prob = 0

        for i in range(1, len(states)):
            if fwd[i][len(observations)] > max_prob:
                max_prob = fwd[i][len(observations)]
                max_index = i
        
        most_likely = []
        cur = len(observations)

        while max_index != 0:
            most_likely.append(states[max_index])
            max_index = backpointers[max_index][cur]
            cur -= 1
        
        return list(reversed(most_likely))

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM Monte Carlo Simulation")
    parser.add_argument("basename", type=str, help="Base name of the HMM files")
    parser.add_argument("--generate", type=int, help="Generate a random sequence of given length")
    parser.add_argument("--forward", type=str, help="Run the forward algorithm on a sequence of observations")
    parser.add_argument("--viterbi", type=str, help="Run the Viterbi algorithm on a sequence of observations")
    args = parser.parse_args()

    basename = sys.argv[1]
    option = sys.argv[2]

    h = HMM()
    h.load(basename)
    valid = ["2,5", "3,4", "4,3", "4,4", "5,5"]


    if args.generate:
        sequence = h.generate(args.generate)
        with open(args.basename + "_sequence.obs", "w") as f:
            f.write(" ".join(sequence.outputseq))

    sequence = Sequence([], [])

    if args.forward:
        with open(args.forward, "r") as f:
            sequence.outputseq = f.readline().strip().split(" ")
            probable = h.forward(sequence)
        if basename == "lander" :
            if probable in valid:
                print(probable)
                print("Safe to land")
            else:
                print(probable)
                print("Not safe to land")
        else:
            print(probable)

    if args.viterbi:
        with open(args.viterbi, "r") as f:
            sequence.outputseq = f.readline().strip().split(" ")
            probable = h.viterbi(sequence)
            print(probable)
        

