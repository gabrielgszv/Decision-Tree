import math
from sklearn.datasets import load_breast_cancer
import random

class Tree:
   def __init__(self, parent=None):
      self.parent = parent
      self.children = []
      self.label = None
      self.classCounts = None
      self.splitFeatureValue = None
      self.splitFeature = None


def printBinaryDecisionTree(root, indentation=""):
   if root.children == []:
      print(f"{indentation}{root.splitFeatureValue}, {root.label} {root.classCounts}")
   else:
      printBinaryDecisionTree(root.children[0], indentation + "\t")

      if indentation == "": 
         print(f"{indentation}{root.splitFeature}")
      else:
         print(f"{indentation}{root.splitFeatureValue}, {root.splitFeature}")

      if len(root.children) == 2:
         printBinaryDecisionTree(root.children[1], indentation + "\t")


def dataToDistribution(data):
   allLabels = [label for (point, label) in data]
   numEntries = len(allLabels)
   possibleLabels = set(allLabels)

   dist = []
   for aLabel in possibleLabels:
      dist.append(float(allLabels.count(aLabel)) / numEntries)

   return dist


def entropy(dist):
   return -sum([p * math.log(p, 2) for p in dist if p > 0])


def splitData(data, featureIndex):
   attrValues = [point[featureIndex] for (point, label) in data]

   for aValue in set(attrValues):
      dataSubset = [(point, label) for (point, label) in data
                    if point[featureIndex] == aValue]

      yield dataSubset


def gain(data, featureIndex):
   entropyGain = entropy(dataToDistribution(data))

   for dataSubset in splitData(data, featureIndex):
      entropyGain -= entropy(dataToDistribution(dataSubset))

   return entropyGain


def homogeneous(data):
   return len(set([label for (point, label) in data])) <= 1


def majorityVote(data, node):
   labels = [label for (pt, label) in data]
   choice = max(set(labels), key=labels.count)
   node.label = choice
   node.classCounts = dict([(label, labels.count(label)) for label in set(labels)])
   return node


def buildDecisionTree(data, root, remainingFeatures):
   if homogeneous(data):
      root.label = data[0][1]
      root.classCounts = {root.label: len(data)}
      return root

   if len(remainingFeatures) == 0:
      return majorityVote(data, root)

   bestFeature = max(remainingFeatures, key=lambda index: gain(data, index))

   if gain(data, bestFeature) == 0:
      return majorityVote(data, root)

   root.splitFeature = bestFeature

   for dataSubset in splitData(data, bestFeature):
      aChild = Tree(parent=root)
      aChild.splitFeatureValue = dataSubset[0][0][bestFeature]
      root.children.append(aChild)

      buildDecisionTree(dataSubset, aChild, remainingFeatures - set([bestFeature]))

   return root


def decisionTree(data):
   return buildDecisionTree(data, Tree(), set(range(len(data[0][0]))))


def classify(tree, point):
   if tree.children == []:
      return tree.label
   else:
      matchingChildren = [child for child in tree.children
         if child.splitFeatureValue == point[tree.splitFeature]]

      if len(matchingChildren) == 0:
         raise Exception("Classify is not able to handle noisy data. Use classify2 instead.")

      return classify(matchingChildren[0], point)

def dictionarySum(*dicts):
    ''' Return a key-wise sum of a list of dictionaries with numeric values. '''
    sumDict = {}
    for aDict in dicts:
        for key in aDict:
            if key in sumDict:
                sumDict[key] += aDict[key]
            else:
                sumDict[key] = aDict[key]
    return sumDict

def classifyNoisy(tree, point):
    ''' Classifica um ponto ruidoso, retornando a contagem das classes. '''
    if tree.children == []:  # Se for um nó folha, retorne as contagens
        return tree.classCounts
    elif point[tree.splitFeature] == '?':  # Valor ausente
        # Combine os resultados de todos os filhos
        dicts = [classifyNoisy(child, point) for child in tree.children]
        return dictionarySum(*dicts)
    else:
        # Procura um filho correspondente
        matchingChildren = [child for child in tree.children
                            if child.splitFeatureValue == point[tree.splitFeature]]
        if len(matchingChildren) == 0:
            # Retorne as contagens do nó atual, se disponíveis
            return tree.classCounts if tree.classCounts else {}
        return classifyNoisy(matchingChildren[0], point)


def classify2(tree, point):
    ''' Classificar dados com ruído, retornando a classe mais provável. '''
    counts = classifyNoisy(tree, point)
    if not counts:  # Se counts está vazio, retorne uma classe padrão ou avise
        return None
    return max(counts.keys(), key=lambda k: counts[k])


def testClassification(data, tree, classifier=classify2):
   actualLabels = [label for point, label in data]
   predictedLabels = [classifier(tree, point) for point, label in data]

   correctLabels = [(1 if a == b else 0) for a, b in zip(actualLabels, predictedLabels)]
   return float(sum(correctLabels)) / len(actualLabels)




# Carregar o dataset breast_cancer e formatar os dados
def loadbreast_cancerDataset():
   breast_cancer = load_breast_cancer()
   data = list(zip(breast_cancer.data.tolist(), breast_cancer.target.tolist()))
   return data, breast_cancer.target_names

def load_data_from_file(file_path):
    """
    Lê um arquivo CSV onde cada linha contém 4 características e o rótulo da classe.
    Retorna uma lista de tuplas (características, classe).
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Remove espaços e quebras de linha
            line = line.strip()
            if not line:
                continue
            
            # Divide os valores por vírgula
            parts = line.split(',')
            features = list(map(float, parts[2:]))  # As 4 primeiras são características
            label = parts[1]  # A última é o rótulo da classe
            data.append((features, label))
    return data

def separar_treino_teste(dados, proporcao_teste=0.2):
    """
    Separa os dados em treino e teste de forma aleatória.

    :param dados: Lista de dados
    :param proporcao_teste: Proporção de dados para teste (restante será treino)
    :return: (dados_treino, dados_teste)
    """
    # Embaralha os dados antes de separar
    dados_embaralhados = random.sample(dados, len(dados))  # Embaralha os dados
    quantidade_teste = int(len(dados) * proporcao_teste)
    
    # Divide os dados em teste e treino
    dados_teste = dados_embaralhados[:quantidade_teste]
    dados_treino = dados_embaralhados[quantidade_teste:]
    
    return dados_treino, dados_teste


def classify_and_compare(tree, data):
    """
    Classifica os dados usando a árvore de decisão e compara o resultado com o rótulo real.
    """
    correct = 0
    total = len(data)
    for features, true_label in data:
        
        predicted_label = classify(tree, features)
        
        # Converte a previsão numérica de volta para o nome da classe
        breast_cancer = load_breast_cancer()
        predicted_label_name = breast_cancer.target_names[predicted_label]
        
        # Exibe o resultado para cada linha
        print(f"Entrada: {features} => Previsto: {predicted_label_name}, Real: {true_label}")
        
        # Conta os acertos
        if predicted_label_name == true_label:
            correct += 1
    
    # Exibe a acurácia final
    accuracy = correct / total
    print(f"\nAcurácia: {accuracy:.2%}")

# Testar o código com o dataset breast_cancer
target_names = loadbreast_cancerDataset()[1]
print(target_names)
data = load_data_from_file('wdbc.data')

treino, teste = separar_treino_teste(data, proporcao_teste=0.5)

#Criar uma árvore com o conjunto de treino   
tree = decisionTree(treino)

#Calcular a acurácia
accuracy = testClassification(teste, tree)
print(f"Acurácia da árvore de decisão no dataset breast_cancer: {accuracy:.2f}")

#Exemplo para ver se ta retornando certo
exemplo = [12.75,16.7,82.51,493.8,0.1125,0.1117,0.0388,0.02995,0.212,0.06623,0.3834,1.003,2.495,28.62,0.007509,0.01561,0.01977,0.009199,0.01805,0.003629,14.45,21.74,93.63,624.1,0.1475,0.1979,0.1423,0.08045,0.3071,0.08557]
print(classify2(tree, exemplo))


