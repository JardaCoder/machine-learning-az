import Orange
from Orange.classification import MajorityLearner
from Orange.evaluation.testing import TestOnTestData
from Orange.evaluation import CA
from collections import Counter

base_credit = Orange.data.Table('credit_data_regras.csv')

print(base_credit.domain)

majority = MajorityLearner()

previsions = TestOnTestData(base_credit, base_credit,  [majority])

print(CA(previsions))

print(Counter(str(register.get_class()) for register in base_credit))

