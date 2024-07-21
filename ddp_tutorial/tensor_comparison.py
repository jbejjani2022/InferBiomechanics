import torch

def compare(t1, t2):
    print(f't1 = t2? {torch.equal(t1, t2)}')

t1 = torch.zeros(5)
print(f't1 = {t1}')

t2 = torch.zeros(5)
print(f't2 = {t2}')

t3 = torch.tensor([4,3,2])
print(f't3 = {t3}')
# t3 = tuple(sorted(t3))
# print(f't3 = {t3}')

# t1 = tuple(sorted(t1))
# t2 = tuple(sorted(t2))

accumulated = []
gathered = [tuple(t1.tolist()), tuple(t2.tolist()), tuple(t3.tolist())]
accumulated.extend(gathered)
print(f'All batches: {accumulated}')
print(f'Set of batches: {set(accumulated)}')


print(tuple(t3), tuple(t3.tolist()))
# compare(t1, t2)

s = set()
s.add(tuple(t1.tolist()))
s.add(tuple(t2.tolist()))
print(s)