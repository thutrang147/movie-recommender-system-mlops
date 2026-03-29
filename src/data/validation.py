# File: src/data/validation.py
import pandas as pd
import os
from pydantic import BaseModel, Field, ValidationError

# 1. Define Data Contracts
class RatingContract(BaseModel):
    user_id: int = Field(..., gt=0)
    movie_id: int = Field(..., gt=0)
    rating: int = Field(..., ge=1, le=5)
    timestamp: int = Field(..., gt=0)

class MovieContract(BaseModel):
    movie_id: int = Field(..., gt=0)
    title: str = Field(..., min_length=1)
    genres: str = Field(..., min_length=1)

class UserContract(BaseModel):
    user_id: int = Field(..., gt=0)
    gender: str = Field(..., pattern="^(M|F)$") # Only accept M or F
    age: int = Field(..., gt=0)
    occupation: int = Field(..., ge=0, le=20)
    zip_code: str = Field(..., min_length=1)

# 2. Processing & Convert
def validate_and_convert():
    raw_dir = 'data/raw/'
    processed_dir = 'data/processed/'
    os.makedirs(processed_dir, exist_ok=True)

    print("Starting Validation & Convert MovieLens 1M...\n")

    # Processing MOVIES
    print("1. Processing movies.dat...")
    df_movies = pd.read_csv(f'{raw_dir}movies.dat', sep='::', engine='python', 
                            header=None, names=['movie_id', 'title', 'genres'], encoding='latin-1')
    
    valid_movies = []
    # Scan row by row to validate
    for index, row in df_movies.iterrows():
        try:
            valid_movies.append(MovieContract(**row.to_dict()).model_dump())
        except ValidationError:
            print(f"  [!] Skip invalid movie at row {index}: {row['title']}")
            
    df_movies_clean = pd.DataFrame(valid_movies)
    df_movies_clean.to_parquet(f'{processed_dir}movies.parquet', index=False)
    print(f"✅ Saved {len(df_movies_clean)} clean movies into {processed_dir}movies.parquet from {len(df_movies)}\n")


    # Processing USERS
    print("2. Processing users.dat...")
    df_users = pd.read_csv(f'{raw_dir}users.dat', sep='::', engine='python', 
                           header=None, names=['user_id', 'gender', 'age', 'occupation', 'zip_code'])
    
    valid_users = []
    for index, row in df_users.iterrows():
        try:
            valid_users.append(UserContract(**row.to_dict()).model_dump())
        except ValidationError:
            pass # Silently skip invalid user
            
    df_users_clean = pd.DataFrame(valid_users)
    df_users_clean.to_parquet(f'{processed_dir}users.parquet', index=False)
    print(f"✅ Saved {len(df_users_clean)} clean users into {processed_dir}users.parquet from {len(df_users)}\n")


    # Processing RATINGS
    print("3. Processing ratings.dat (Around 1 million rows)")
    df_ratings = pd.read_csv(f'{raw_dir}ratings.dat', sep='::', engine='python', 
                             header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
    
    # With 1 million rows, running the Pydantic for loop will be a bit slow. 
    # In a real environment, we combine the power of Pandas Vectorize for large data
    initial_len = len(df_ratings)
    df_ratings_clean = df_ratings[
        (df_ratings['rating'] >= 1) & (df_ratings['rating'] <= 5) &
        (df_ratings['user_id'] > 0) & (df_ratings['movie_id'] > 0)
    ]
    
    dropped = initial_len - len(df_ratings_clean)
    df_ratings_clean.to_parquet(f'{processed_dir}ratings.parquet', index=False)
    print(f"✅ Cleaned! Dropped {dropped} invalid ratings.")
    print(f"✅ Saved {len(df_ratings_clean)} clean ratings into {processed_dir}ratings.parquet\n")

    print("🎉 ALL DATA HAS BEEN VALIDATED AND READY FOR THE NEXT STEP!")

if __name__ == "__main__":
    validate_and_convert()