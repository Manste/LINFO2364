import pandas as pd
import matplotlib.pyplot as plt

dataApMush = pd.read_csv("Performance/apriorimushroom.csv")
dataEcMush = pd.read_csv("Performance/eclatmushroom.csv")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(dataApMush.minFrequency,dataApMush.duration,'-gD', label='Apriori')
plt.plot(dataEcMush.minFrequency,dataEcMush.duration,'-rD', label='Eclat')
plt.title("Computing time with mushroom.dat")
plt.xlabel('Frequency')
plt.ylabel('Time [s]')
plt.legend()
plt.xlim(max(dataApMush.minFrequency),min(dataApMush.minFrequency))

plt.subplot(1, 2, 2)
plt.plot(dataApMush.minFrequency,dataApMush.Peak,'-gD', label='Apriori')
plt.plot(dataEcMush.minFrequency,dataEcMush.Peak,'-rD', label='Eclat')
plt.title("Memory Peak with mushroom.dat")
plt.xlabel('Frequency')
plt.ylabel('Memory [MB]')
plt.legend()
plt.xlim(max(dataApMush.minFrequency),min(dataApMush.minFrequency))
plt.savefig('Performance/mushroom.jpg')
plt.show()

#**************************************************************

dataApChess = pd.read_csv("Performance/apriorichess.csv")
dataEcChess = pd.read_csv("Performance/eclatchess.csv")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(dataApChess.minFrequency,dataApChess.duration, '-gD', label='Apriori')
plt.plot(dataEcChess.minFrequency,dataEcChess.duration,'-rD', label='Eclat')
plt.title("Computing time with chess.dat")
plt.xlabel('Frequency')
plt.ylabel('Time [s]')
plt.legend()
plt.xlim(max(dataApChess.minFrequency),min(dataApChess.minFrequency))

plt.subplot(1, 2, 2)
plt.plot(dataApChess.minFrequency,dataApChess.Peak, '-gD', label='Apriori')
plt.plot(dataApChess.minFrequency,dataEcChess.Peak,'-rD', label='Eclat')
plt.title("Memory Peak with chess.dat")
plt.xlabel('Frequency')
plt.ylabel('Memory [MB]')
plt.legend()
plt.xlim(max(dataApChess.minFrequency),min(dataApChess.minFrequency))
plt.savefig('Performance/chess.jpg')
plt.show()