"""Bounded, full-precision text batching for large generated Stim circuits."""
import stim


class CircuitAccumulator:
    """Same append ordering and double values, fewer Python/C++ crossings."""
    def __init__(self):
        self.circuit=stim.Circuit();self.lines=[]

    def flush(self):
        if self.lines:
            self.circuit+=stim.Circuit('\n'.join(self.lines));self.lines=[]

    def append(self,instruction,targets=None,probabilities=None):
        if isinstance(instruction,str):
            if instruction!='PAULI_CHANNEL_1':raise ValueError('unexpected batched instruction')
            self.lines.append('PAULI_CHANNEL_1('+','.join(repr(float(v)) for v in probabilities)+') '+ ' '.join(str(int(t)) for t in targets))
        else:
            if getattr(instruction,'tag',''):
                self.flush();self.circuit.append(instruction);return
            line=str(instruction);args=instruction.gate_args_copy()
            # Stim's display string rounds probabilities to six significant
            # figures. Replace the argument text with full precision before parsing.
            if args:
                lo=line.index('(');hi=line.index(')',lo)
                line=line[:lo+1]+','.join(repr(float(v)) for v in args)+line[hi:]
            self.lines.append(line)
        if len(self.lines)>=4096:self.flush()

    def finish(self):
        self.flush();return self.circuit
