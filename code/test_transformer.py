from typing_extensions import Annotated
from verifier import *
from helper_fun import spu
import numpy as np


def test_spu_transformer_soundness():
    number_of_tests = 20000
    select_transformer = "midpoint2"
    print(f"Testing soundness of spu transformer \"{select_transformer}\" for {number_of_tests} examples")

    for i in range(number_of_tests):
        if i % (number_of_tests/10) == 0:
            print(f"{int(i/(number_of_tests/100))}%...", end=' ')
        l, u = np.random.randn(), np.random.randn()
        if l > u:
            l, u = u, l
        #print(l,u)

        abs_x = ANode(l, u)
        transformed_x = abs_x.spu_transformer(select_transformer)
        #print(transformed_x)
        for x in np.linspace(l, u, 1000):
            spu_x = spu(x)
            assert spu_x > transformed_x.lower_bound - 1e-5 # test if it is larger than lower bound
            assert spu_x < transformed_x.upper_bound + 1e-5 # test if it is smaller than upper bound
            abs_x.set_concrete_val(x)
            back_sub_lbound = transformed_x.l_relation_constraint.eval()
            back_sub_ubound = transformed_x.u_relation_constraint.eval()
            try:
                assert spu_x > back_sub_lbound - 1e-5
            except:
                print(f"\nInterval: [{l},{u}]\tSPU: {spu_x}   TransformerLower: {back_sub_lbound}")
                #return

            try:    
                assert spu_x < back_sub_ubound + 1e-5
            except:
                print(f"\nInterval: [{l},{u}]\tSPU: {spu_x}   TransformerUpper: {back_sub_ubound}")
                return

    print("Done. No unsound examples found.")

def compare_spu_transformer_precision(transformer1:str, transformer2:str):
    number_of_tests = 50000
    print(f"Comparing precision of spu transformers for {number_of_tests} examples")

    equal, t1_better, t2_better = 0, 0, 0

    for i in range(number_of_tests):
        if i % (number_of_tests/10) == 0:
            print(f"{int(i/(number_of_tests/100))}%...", end=' ')
        l, u = np.random.randn(), np.random.randn()
        if l > u:
            l, u = u, l
        
        abs_x = ANode(l, u)
        t1 = abs_x.spu_transformer(transformer1)
        t2 = abs_x.spu_transformer(transformer2)
        t1_lbound = t1.l_relation_constraint.backwards_sub(False)
        t1_ubound = t1.u_relation_constraint.backwards_sub(True)
        t2_lbound = t2.l_relation_constraint.backwards_sub(False)
        t2_ubound = t2.u_relation_constraint.backwards_sub(True)

        if t1_lbound >= t2_lbound and t1_ubound <= t2_ubound and (t1_lbound > t2_lbound or t1_ubound < t2_ubound):
            t1_better += 1
        elif t1_lbound <= t2_lbound and t1_ubound >= t2_ubound and (t1_lbound < t2_lbound or t1_ubound > t2_ubound):
            t2_better += 1
        else:
            equal += 1
    
    print()
    print(f"{transformer1} better:\t{t1_better} ({round(t1_better/number_of_tests*100, 2)}%)")
    print(f"{transformer2} better:\t{t2_better} ({round(t2_better/number_of_tests*100, 2)}%)")
    print(f"Transformers not comparable:\t{equal} ({round(equal/number_of_tests*100, 2)}%)")


def example_linear_transformer():
    # Example of the first linear layer from the lecture slides, slide 8
    x1 = ANode(-1, 1)
    x2 = ANode(-1, 1)
    print(x1)
    print(x2)
    
    linear_comb_3 = Relational_Constraint([(1,x1), (1,x2)])
    x3 = ANode.from_constraint(linear_comb_3)
    print(x3)

    linear_comb_4 = Relational_Constraint([(1,x1), (-1,x2)])
    x4 = ANode.from_constraint(linear_comb_4)
    print(x4)

def test_back_sub():
    # taken from exercise 6
    x1 = ANode(1,2)
    x2 = ANode(-1,0)
    x3 = ANode(0,3)
    x4 = ANode.from_constraint(Relational_Constraint([(1, x1),(-1,x2),(2, x3)], 1/2))
    x5 = ANode.from_constraint(Relational_Constraint([(1/2, x1),(-1/2,x2),(1, x3)], -3/2))
    x6 = ANode.from_constraint(Relational_Constraint([(2,x4),(-3,x5)],1.0))
    #x7 = ANode(0,1,Relational_Constraint([]), Relational_Constraint([(1/3,x4)], 2/3))
    #x8 = ANode.from_constraint(Relational_Constraint([(1,x5),(1,x6)]))
    #x9 = ANode.from_constraint(Relational_Constraint([(-1,x5),(1,x6)]))
    
    for x in [x1, x2, x3, x4, x5, x6]:
        print(x)
        pass
    
    print(f"x6 with back_sub: [{x6.l_relation_constraint.backwards_sub(False)},{x6.u_relation_constraint.backwards_sub(True)}]")
    #print(f"x8 with back_sub: [{x8.l_relation_constraint.backwards_sub(False)},{x8.u_relation_constraint.backwards_sub(True)}]")
    #print(f"x7-x8 >= {Relational_Constraint([(1,x7),(-1,x8)]).backwards_sub(False)}")
    

#example_linear_transformer()
#test_spu_transformer_soundness()
# compare_spu_transformer_precision("own1", "own2")
test_back_sub()