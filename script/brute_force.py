import numpy as np
import pandas as pd
import itertools, math
from utils import PerplexityCalculator

def main():
    df = pd.read_csv("../input/sample_submission.csv")
    words = df.loc[0,"text"].split()
    
    all_permutations = list( itertools.permutations(words) )[::-1]
    PERM_CT = math.factorial(10)
    
    print(f"There are {PERM_CT} possible permutations for first sample!")

    # LOAD GEMMA SCORER
    scorer = PerplexityCalculator('../model/gemma-2-9b')
    
    # 時間計測
    import time
    start = time.time()
    
    # 10回計算する
    for i in range(10):
        m = scorer.get_perplexity(df.loc[0, "text"], batch_size=1)
        print(f"The perplexity of the first sample without permutation is {m:.2f}.")
    # m = scorer.get_perplexity(df.loc[0, "text"], batch_size=1)
    # print(f"The perplexity of the first sample without permutation is {m:.2f}.")
    
    elapsed = time.time() - start
    print(f"Elapsed time: {elapsed:.2f} seconds.")
    

    # 設定    
    BATCH_SIZE = 64

    best_score = 1e6
    best_text = ""

    # 全ての計算を全探索する
    perms = []
    for i, p in enumerate(all_permutations):
        # スペース区切りの文字列に変換
        perms.append(" ".join(p))
        
        # バッチサイズ分まとめて計算
        if (len(perms)==BATCH_SIZE) | (i==PERM_CT-1): 
            # スコア計算して更新したら記録
            p = scorer.get_perplexity(perms, batch_size=BATCH_SIZE)
            if np.min(p) < best_score:
                best_score = np.min(p)
                best_text = perms[ np.argmin(p) ]
                print( f"New best = {best_score} with '{best_text}'" )
            perms = []
        # 進捗表示
        if i%10_000==0: 
            print(f"Completed computing {i} perplexities.")
        if best_score < 475: 
            print("Stopping early because we found optimal!")
            break

    print("Done.")

if __name__ == '__main__':
    main()