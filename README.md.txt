# Classificação de Sentimentos usando NLTK

## Problema
Classificação de sentimentos é uma técnica de Processamento de Linguagem Natural que identifica se uma frase é subjetiva ou objetiva. É usada em redes sociais, análise de produtos, filmes etc.  

O modelo enfrenta desafios como sarcasmo, ironia, negação, ambiguidade e dependência de dataset pequeno.

## Método
- Biblioteca utilizada: **NLTK** em Python  
- Dataset: **Subjectivity** (100 frases subjetivas e 100 objetivas)  
- Modelo: **Naive Bayes** com **unigramas**  
- Experimentos: 9 frases de teste com sentimentos positivos, negativos, negações e ironia  

## Resultados
| Frase | Resultado |
|-------|----------|
| This car is beautiful | obj |
| This car is horrible | obj |
| I like this perfume | subj |
| I hate this perfume | obj |
| I love it | obj |
| This series is terrible | obj |
| This series is amazing | obj |
| Great, the bread is gone | obj |
| This sneaker is not good | obj |

O modelo classificou a maioria das frases como objetiva, mesmo quando eram opiniões. Mostrou dificuldade com negações e sarcasmo.

## Conclusões
- Dataset pequeno limita aprendizado  
- Unigramas não captam negações, sarcasmo ou contexto  
- Frases curtas ou genéricas podem ser classificadas errado  

Possíveis melhorias:
- Usar mais dados de treino
- Modelos avançados como **VADER** ou **BERT**
- Usar bigramas ou embeddings

## Referências
1. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O’Reilly Media.  
2. NLTK Documentation – Sentiment Analysis HowTo. Disponível em: [https://www.nltk.org/howto/sentiment.html](https://www.nltk.org/howto/sentiment.html)  
3. Hutto, C.J., & Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text*.