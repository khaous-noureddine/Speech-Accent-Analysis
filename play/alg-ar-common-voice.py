import os
import sys
import requests
import tarfile
import pandas as pd

def download_and_analyze(url):
    target_dir = "/nfs/homes/noureddine.khaous/napster/accented-speech-recognition/play"
    os.makedirs(target_dir, exist_ok=True)
    
    validated_path = os.path.join(target_dir, "validated.tsv")
    durations_path = os.path.join(target_dir, "clip_durations.tsv")
    
    if not (os.path.exists(validated_path) and os.path.exists(durations_path)):
        print("Streaming Common Voice archive to extract metadata only...")
        
        # Stream the download
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        found_validated = False
        found_durations = False
        
        # Read the tar stream on the fly
        with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
            for member in tar:
                if member.name.endswith("validated.tsv"):
                    print(f"Found {member.name}, extracting...")
                    with open(validated_path, "wb") as f_out:
                        f_out.write(tar.extractfile(member).read())
                    found_validated = True
                    
                elif member.name.endswith("clip_durations.tsv"):
                    print(f"Found {member.name}, extracting...")
                    with open(durations_path, "wb") as f_out:
                        f_out.write(tar.extractfile(member).read())
                    found_durations = True
                    
                # Kill the stream once we have both files to avoid downloading audio
                if found_validated and found_durations:
                    print("Metadata secured. Aborting the rest of the download.")
                    break
                    
        if not (found_validated and found_durations):
            print("Error: Could not find the required .tsv files in the stream.")
            sys.exit(1)
    else:
        print("Metadata files already exist in the play folder. Skipping download.")

    print("\nLoading and analyzing data...")
    df_meta = pd.read_csv(validated_path, sep='\t', low_memory=False)
    df_dur = pd.read_csv(durations_path, sep='\t')
    
    if 'path' in df_meta.columns:
        df_meta = df_meta.rename(columns={'path': 'clip'})
        
    df = pd.merge(df_meta, df_dur, on='clip', how='inner')
    algerian_df = df[df['accents'].str.contains('Algerian', case=False, na=False)]
    
    total_hours = algerian_df['duration[ms]'].sum() / (1000 * 60 * 60)
    
    print("-" * 30)
    print(f"Total Algerian clips found: {len(algerian_df)}")
    print(f"Total hours of Algerian Arabic: {total_hours:.3f} hours")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python alg-ar-common-voice.py '<MOZILLA_DOWNLOAD_URL>'")
        sys.exit(1)
        
    download_and_analyze(sys.argv[1])