from pytubefix import YouTube
url = "https://www.youtube.com/watch?v=EYEyYpLpsFs"
#url = "https://www.youtube.com/watch?v=94j_clHMTNM"
#url = "https://www.youtube.com/shorts/oFFBB1faAjs?feature=share"
#url = "https://www.youtube.com/watch?v=hZTl83iopzY&t=23s"
#url = "https://www.youtube.com/watch?v=M-m83y54070&t=13s"
#url = "https://www.youtube.com/watch?v=XBdUrmf4-oQ"
yt = YouTube(url)

print(f" Downloading: {yt.title}")
yt.streams.get_highest_resolution().download(
    output_path="C:/Users/88690/Desktop/Dissertation/paddleOCR"
)
print("Download complete.")
