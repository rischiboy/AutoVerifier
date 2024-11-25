import argparse
from typing import List, Tuple
import torch
from helper_fun import sigmoid, spu, spu_deriv
from networks import FullyConnected

DEVICE = 'cpu'
INPUT_SIZE = 28
select_transformer_global = "own1"

class ANode:
    name_ind = 1

    def __init__(self, lower_bound:float, upper_bound:float, l_relation_const=None, u_relation_const=None):
        """
        Creates a new abstract node. It can be instantiated using only the interval (lower & upper bound),
        the relational constraint will be automatically set. To create a ANode from a linear layer consider
        using ANode.from_constraint and for spu use self.spu_transformer
        """
        self.name = f"x{ANode.name_ind}"
        ANode.name_ind += 1
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if u_relation_const == None:
            self.u_relation_constraint = Relational_Constraint([], upper_bound)
        else:
            self.u_relation_constraint:Relational_Constraint = u_relation_const
        if l_relation_const == None:
            self.l_relation_constraint = Relational_Constraint([], lower_bound)
        else:
            self.l_relation_constraint:Relational_Constraint = l_relation_const

    def back_sub(self, compute_upper):
        """
        Evaluate the bound for this node using backwards substitution
        @param compute_upper - Boolean wheter to compute the upper or lower bound
        """
        if compute_upper:
            return self.u_relation_constraint
        else:
            return self.l_relation_constraint

    def get_bound(self, compute_upper):
        if compute_upper:
            return self.upper_bound
        else:
            return self.lower_bound

    def spu_transformer(self, select_transformer=None):
        """
        Returns a new node which is the abstract representation of spu(self).
        @param select_transformer - optional string indicating which transformer should be used. Allows for easier switching between different transformers. If it is None, select_transformer_global is used.
        """
        if select_transformer is None:
            select_transformer = select_transformer_global
        transformers = {
            "naive": transformer_naive,
            "sigmoid": transformer_sigmoid,
            "own1": transformer_own1,
            "own2": transformer_own2,
            "midpoint1": transformer_midpoint1,
            "midpoint1b": transformer_midpoint1b,
            "midpoint2": transformer_midpoint2,
        }
        return transformers[select_transformer](self)

    def set_concrete_val(self, value):
        """Used for testing purposes"""
        self.concret_val = value

    def print_bounds(self):
        print(f"ANode {self.name}:\t[{self.lower_bound}, {self.upper_bound}]")

    def __str__(self):
        res = f"ANode {self.name}:\n\t"
        res += self.name + " >= " + str(self.l_relation_constraint) + "\n\t"
        res += self.name + " <= " + str(self.u_relation_constraint) + "\n\t"
        res += self.name + f" in [{self.lower_bound}, {self.upper_bound}]\n"
        return res

    @staticmethod
    def from_constraint(rel_constraint:'Relational_Constraint'):
        """
        Creates an abstract node from a constraint. Can be used as the abstract linear transformer.
        """
        return ANode(rel_constraint.compute_bound(False), rel_constraint.compute_bound(True), rel_constraint, rel_constraint)

class Relational_Constraint:
    def __init__(self, weighted_nodes:List[Tuple[float, ANode]], bias:float=0):
        """
        Creates a relational constraint, i.e. a linear combination of nodes + a bias
        @param weighted_nodes - a list of weights and their nodes, each weight w and node x given as a tuple (w, x)
        @param bias - the bias term
        """
        self.weighted_nodes = weighted_nodes
        self.bias = bias

    def compute_bound(self, compute_upper):
        """
        Computes the bound of this constraint using the bounds of the nodes
        @param compute_upper - Boolean whether to compute the upper or lower bound
        """
        res = self.bias
        for w, node in self.weighted_nodes:
            if w > 0:
                res += w * node.get_bound(compute_upper)
            else:
                res += w * node.get_bound(not compute_upper)
        return res

    def backwards_sub(self, compute_upper) -> float:
        """
        Computes the bound of this constraint using backwards propagation
        @param compute_upper - Boolean wheter to compute the upper or lower bound
        """
        #print(self)
        new_constraint = Relational_Constraint([], self.bias)
        for w, node in self.weighted_nodes:
            if w == 0:
                continue
            elif w < 0:
                compute_upper_instance = not compute_upper
            else:
                compute_upper_instance = compute_upper
            constraint = node.back_sub(compute_upper_instance)
            new_constraint.combine_and_simplify(w, constraint)
        if new_constraint.weighted_nodes:
            return new_constraint.backwards_sub(compute_upper)
        else:
            return new_constraint.bias

    def combine_and_simplify(self, weight:float, other:"Relational_Constraint"):
        """
        Combines self with another constraint, simplifying it as much as possible.
        @param weight - constant with which the weights and bias of 'other' will be multiplied with
        @param other - Relational_Constraint which should be combined into self
        """
        self.bias += weight * other.bias
        for w1, node1 in other.weighted_nodes:
            if w1 == 0:
                continue
            to_append = (weight * w1, node1)
            for tuple in self.weighted_nodes:
                w2, node2 = tuple
                if node1 == node2:
                    self.weighted_nodes.remove(tuple)
                    new_w = w2 + weight * w1
                    if new_w == 0:
                        to_append = None
                    else:
                        to_append = (new_w, node1)
                    break
            if to_append != None:
                self.weighted_nodes.append(to_append)
        return self

    def eval(self):
        """
        Evaluates this constraint with given concrete values (note: each node should have a concrete value set)
        """
        res = self.bias
        for w, node in self.weighted_nodes:
            res += w * node.concret_val
        return res

    def __str__(self) -> str:
        res = str(self.bias)
        for w, node in self.weighted_nodes:
            res += f" + {w} * {node.name}"
        return res

def transformer_naive(x:ANode):
    """A naive transformer that looses all relational constraint"""
    return ANode(-0.5, max(sigmoid(-x.lower_bound) - 1, x.upper_bound**2 - 0.5))

def transformer_sigmoid(x:ANode):
    """Using Sigmoid Abstract Transformer"""
    l_bound = spu(x.lower_bound)
    u_bound = spu(x.upper_bound)
    if (x.lower_bound == x.upper_bound):
        l_rel = Relational_Constraint([], l_bound)
        u_rel = Relational_Constraint([], l_bound)
    else:
        lambda_val = (u_bound-l_bound)/(x.upper_bound-x.lower_bound)
        lambda_prime_val = min(spu_deriv(x.lower_bound), spu_deriv(x.upper_bound))

        if x.lower_bound > 0:
            l_rel = Relational_Constraint([(lambda_val, x)], l_bound - lambda_val*x.lower_bound)
        else:
            l_rel = Relational_Constraint([(lambda_prime_val, x)], l_bound - lambda_prime_val*x.lower_bound)
            
        if x.upper_bound <= 0:
            u_rel = Relational_Constraint([(lambda_val, x)], u_bound - lambda_val*x.upper_bound)
        else:
            u_rel = Relational_Constraint([(lambda_prime_val, x)], u_bound - lambda_prime_val*x.upper_bound)
    
    return ANode(l_bound, u_bound, l_rel, u_rel)
        

def transformer_own1(x:ANode):
    """A self written transformer. Uses the derivatives of the lower bound for relational constraints."""
    l_bound = spu(x.lower_bound)
    u_bound = spu(x.upper_bound)
    if (x.lower_bound == x.upper_bound):
        l_rel = Relational_Constraint([], l_bound)
        u_rel = Relational_Constraint([], l_bound)
    else:
        lambda_val = (u_bound-l_bound)/(x.upper_bound-x.lower_bound)
        l_deriv = spu_deriv(x.lower_bound)

        # lower_bound > 0
        l_rel = Relational_Constraint([(l_deriv, x)], l_bound - l_deriv*x.lower_bound)
        u_rel = Relational_Constraint([(lambda_val, x)], l_bound - lambda_val*x.lower_bound)

        if x.upper_bound < 0:
            l_rel, u_rel = u_rel, l_rel
            l_bound, u_bound = u_bound, l_bound
        elif x.lower_bound < 0:
            if l_deriv > lambda_val:
                 # choose the max between l_derivative and lambda_val to be sound. lambda_val is used in u_rel previously, so update it if necessary.
                u_rel = l_rel
            l_rel = Relational_Constraint([], -0.5)
            u_bound = max(l_bound, u_bound)
            l_bound = -0.5

    return ANode(l_bound, u_bound, l_rel, u_rel)

def transformer_own2(x:ANode):
    """A self written transformer. Uses the derivatives of the lower bound for relational constraints."""
    l_bound = spu(x.lower_bound)
    u_bound = spu(x.upper_bound)
    if (x.lower_bound == x.upper_bound):
        l_rel = Relational_Constraint([], l_bound)
        u_rel = Relational_Constraint([], l_bound)
    else:
        lambda_val = (u_bound-l_bound)/(x.upper_bound-x.lower_bound)
        lambda_to_zero_val = (-0.5 - l_bound)/(-x.lower_bound)
        l_deriv = spu_deriv(x.lower_bound)
        u_deriv = spu_deriv(x.upper_bound)
        
        def l_line(delta):
            return [(delta, x)], l_bound - delta*x.lower_bound
        
        def u_line(delta):
            return [(delta, x)], u_bound - delta*x.upper_bound

        # lower_bound > 0
        l1, l2 = l_line(l_deriv)
        l_rel = Relational_Constraint(l1, l2)
        l1, l2 = l_line(lambda_val)
        u_rel = Relational_Constraint(l1, l2)

        if x.upper_bound < 0:
            l_rel, u_rel = u_rel, l_rel
            l_bound, u_bound = u_bound, l_bound
        elif x.lower_bound < 0:
            if l_deriv > lambda_val:
                # choose the max between l_derivative and lambda_val to be sound. lambda_val is used in u_rel previously, so update it if necessary.
                u_rel = l_rel

            u_bound = max(l_bound, u_bound)
            l_bound = -0.5
            if (x.lower_bound > -1 and x.upper_bound > 2):
                l1, l2 = u_line(u_deriv)
                l_rel = Relational_Constraint(l1, l2)
            elif (x.upper_bound < 1):
                l1, l2 = l_line(lambda_to_zero_val)
                l_rel = Relational_Constraint(l1, l2)
            else:
                l_rel = Relational_Constraint([], -0.5)


    return ANode(l_bound, u_bound, l_rel, u_rel)

def transformer_midpoint1(x:ANode):
    """A self written transformer. Uses the derivatives of the lower bound for relational constraints."""
    l_bound = spu(x.lower_bound)
    u_bound = spu(x.upper_bound)
    if (x.lower_bound == x.upper_bound):
        l_rel = Relational_Constraint([], l_bound)
        u_rel = Relational_Constraint([], l_bound)
    else:
        lambda_val = (u_bound-l_bound)/(x.upper_bound-x.lower_bound)
        l_deriv = spu_deriv(x.lower_bound)
        midpoint = (x.upper_bound - x.lower_bound)/2.
        midpoint_deriv = spu_deriv(midpoint)

        # lower_bound > 0
        l_rel = Relational_Constraint([(midpoint_deriv, x)], spu(midpoint) - midpoint_deriv*midpoint)
        u_rel = Relational_Constraint([(lambda_val, x)], l_bound - lambda_val*x.lower_bound)

        if x.upper_bound < 0:
            l_rel, u_rel = u_rel, l_rel
            l_bound, u_bound = u_bound, l_bound
        elif x.lower_bound < 0:
            if l_deriv > lambda_val:
                 # choose the max between l_derivative and lambda_val to be sound. lambda_val is used in u_rel previously, so update it if necessary.
                u_rel = l_rel
            l_rel = Relational_Constraint([], -0.5)
            u_bound = max(l_bound, u_bound)
            l_bound = -0.5

    return ANode(l_bound, u_bound, l_rel, u_rel)

def transformer_midpoint1b(x:ANode):
    """Sound variant of the midpoint1 transformer. But achieves worse results."""
    l_bound = spu(x.lower_bound)
    u_bound = spu(x.upper_bound)
    if (x.lower_bound == x.upper_bound):
        l_rel = Relational_Constraint([], l_bound)
        u_rel = Relational_Constraint([], l_bound)
    else:
        lambda_val = (u_bound-l_bound)/(x.upper_bound-x.lower_bound)
        l_deriv = spu_deriv(x.lower_bound)
        midpoint = (x.upper_bound + x.lower_bound)/2.
        midpoint_deriv = spu_deriv(midpoint)

        # lower_bound > 0
        l_rel = Relational_Constraint([(midpoint_deriv, x)], spu(midpoint) - midpoint_deriv*midpoint)
        u_rel = Relational_Constraint([(lambda_val, x)], l_bound - lambda_val*x.lower_bound)

        if x.upper_bound < 0:
            l_rel, u_rel = u_rel, l_rel
            l_bound, u_bound = u_bound, l_bound
        elif x.lower_bound < 0:
            if l_deriv > lambda_val:
                # choose the max between l_derivative and lambda_val to be sound. lambda_val is used in u_rel previously, so update it if necessary.
                u_rel = Relational_Constraint([(l_deriv, x)], l_bound - l_deriv*x.lower_bound)
            l_rel = Relational_Constraint([], -0.5)
            u_bound = max(l_bound, u_bound)
            l_bound = -0.5

    return ANode(l_bound, u_bound, l_rel, u_rel)

def transformer_midpoint2(x:ANode):
    """A self written transformer. Uses the derivatives of the lower bound for relational constraints."""
    l_bound = spu(x.lower_bound)
    u_bound = spu(x.upper_bound)
    if (x.lower_bound == x.upper_bound):
        l_rel = Relational_Constraint([], l_bound)
        u_rel = Relational_Constraint([], l_bound)
    else:
        lambda_val = (u_bound-l_bound)/(x.upper_bound-x.lower_bound)
        l_deriv = spu_deriv(x.lower_bound)
        delta_to_zero_val = (-0.5 - l_bound)/(-x.lower_bound)
        # fixed typo in midpoint, idk why it works above
        midpoint = (x.upper_bound + x.lower_bound)/2.
        midpoint_deriv = spu_deriv(midpoint)

        # lower_bound > 0
        l_rel = Relational_Constraint([(midpoint_deriv, x)], spu(midpoint) - midpoint_deriv*midpoint)
        u_rel = Relational_Constraint([(lambda_val, x)], l_bound - lambda_val*x.lower_bound)

        if x.upper_bound < 0:
            l_rel, u_rel = u_rel, l_rel
            l_bound, u_bound = u_bound, l_bound
        elif x.lower_bound < 0:
            if l_deriv > lambda_val:
                 # choose the max between l_derivative and lambda_val to be sound. lambda_val is used in u_rel previously, so update it if necessary.
                u_rel = Relational_Constraint([(l_deriv, x)], l_bound - l_deriv*x.lower_bound)

            # additional checks if triangle would make sense, otherwise take -0.5 as lower bound
            if x.lower_bound < -1.5*x.upper_bound:
                l_rel = Relational_Constraint([(delta_to_zero_val, x)], l_bound - delta_to_zero_val*x.lower_bound)
            elif x.upper_bound > -1.5*x.lower_bound:
                l_rel = Relational_Constraint([(midpoint_deriv, x)], spu(midpoint) - midpoint_deriv*midpoint)
            else:
                l_rel = Relational_Constraint([], -0.5)
            u_bound = max(l_bound, u_bound)
            l_bound = -0.5

    return ANode(l_bound, u_bound, l_rel, u_rel)

def get_model_params(net):
    model_params = net.parameters()
    bias, weights = [], []
    for i, param in enumerate(model_params):
        if i % 2 == 0:
            weights.append(param.tolist())
        else:
            bias.append(param.tolist())
    return weights, bias

def analyze(net, inputs, eps, true_label):
    weights, bias = get_model_params(net)
    net_dims = [len(b) for b in bias]

    flatten_inputs = inputs.view(INPUT_SIZE*INPUT_SIZE)
    eps_vector = eps * torch.ones(INPUT_SIZE*INPUT_SIZE)
    input_lower_bound = flatten_inputs - eps_vector
    input_upper_bound = flatten_inputs + eps_vector

    # Iterate over the different transformers, set verified if any of them verify the network.
    for transformer in ['midpoint1']:
        global select_transformer_global
        select_transformer_global = transformer

        #Stores all ANodes of the Neural Net
        ANode_list: List[List[ANode]] = []
        #Create ANodes for the input layer
        ANode_input = []
        for lb, ub in zip(input_lower_bound,input_upper_bound):
            anode = ANode(max(lb.tolist(),0),min(ub.tolist(),1))
            ANode_input.append(anode)

        ANode_list.append(ANode_input)    
        
        #Create Relational_Constraint and ANodes for the other layers
        lin_layer = 0
        for dims in net_dims:
            layer_weights, layer_bias = weights[lin_layer], bias[lin_layer]
            ANode_layer_list = []

            for i in range(dims):
                layer_node_weights, layer_node_bias = layer_weights[i], layer_bias[i]
                weighted_nodes = list(zip(layer_node_weights,ANode_list[-1]))
                rel_constraint  = Relational_Constraint(weighted_nodes,layer_node_bias)
                anode = ANode.from_constraint(rel_constraint)
                ANode_layer_list.append(anode)
            
            ANode_list.append(ANode_layer_list)
            Spu_layer_list = []
            
            #No SPU layer for the last (output) layer
            if(lin_layer < len(bias)-1):
                for i in range(dims):
                    spu_anode_i = ANode_layer_list[i].spu_transformer()
                    Spu_layer_list.append(spu_anode_i)

                ANode_list.append(Spu_layer_list)

            lin_layer += 1

        #Check whether units of every layers modeled by abstracted nodes
        # assert len(ANode_list) == len(net_dims) + 1 and list(map(len,ANode_list[1:])) == net_dims

        last_layer = ANode_list[-1]
        true_node = last_layer[true_label]
        verified = True
        for i in range(len(last_layer)):
            #print(last_layer[i])
            #last_layer[i].print_bounds()
            if i != true_label:
                node = last_layer[i]
                if true_node.lower_bound > node.upper_bound:
                    # computed bounds suffice to prove property
                    continue

                # true_node > node ===> true_node - node > 0
                rel_constraint = Relational_Constraint([(1,true_node), (-1,node)])
                l_bound_back_sub = rel_constraint.backwards_sub(False)
                if l_bound_back_sub <= 0:
                    verified = False
                    break

        if verified:
            return 1

    return 0


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
    parser.add_argument('--net',
                        type=str,
                        required=True,
                        help='Neural network architecture which is supposed to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    if args.net.endswith('fc1'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif args.net.endswith('fc2'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif args.net.endswith('fc3'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net.endswith('fc4'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif args.net.endswith('fc5'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    else:
        assert False

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
