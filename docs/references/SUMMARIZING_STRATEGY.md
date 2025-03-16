# Reference Materials Documentation

## Video References

### Downloading Video Subtitles
To download subtitles from YouTube videos, we use `yt-dlp` with specific flags to get English subtitles in SRT format:

```bash
yt-dlp --write-subs --write-auto-subs --skip-download --sub-lang en --convert-subs srt "VIDEO_URL"
```

Key flags explained:
- `--write-subs`: Download available subtitles
- `--write-auto-subs`: Download auto-generated subtitles if available
- `--skip-download`: Skip downloading the actual video
- `--sub-lang en`: Only download English subtitles
- `--convert-subs srt`: Convert subtitles to SRT format

### Referenced Videos

1. "Inverted Pendulum Neural Network" (Original Pezzza Implementation)
   - URL: https://www.youtube.com/watch?v=EvV5Qtp_fYg
   - Transcript: `How to train simple AIs [EvV5Qtp_fYg].en.srt`
   - Key Topics: Neural network implementation, training methodology

2. "Reinforcement Learning - My Algorithm vs State of the Art"
   - URL: https://www.youtube.com/watch?v=pJfvPMNPZAU
   - Transcript: `Reinforcement Learning - My Algorithm vs State of the Art [pJfvPMNPZAU].en.srt`
   - Key Topics: RL approaches comparison, algorithm analysis

3. "Double Pendulum Neural Network"
   - URL: https://www.youtube.com/watch?v=9gQQAO4I1Ck
   - Transcript: `How to train simple AIs to balance a double pendulum [9gQQAO4I1Ck].en.srt`
   - Key Topics: Double pendulum control, neural network application

### Transcript Cleanup Process

After downloading the raw transcripts, we follow these steps to create cleaned versions:

1. Extract Content
   - extract the content of the transcript without any additional formatting or summarizing
   - we want this content to be in a plain text format

   Example

```srt
1
00:00:00,080 --> 00:00:02,550

machine learning is a Hot Topic and for

2
00:00:02,550 --> 00:00:02,560
machine learning is a Hot Topic and for
 

3
00:00:02,560 --> 00:00:05,390
machine learning is a Hot Topic and for
good reason it has unlocked the ability

4
00:00:05,390 --> 00:00:05,400
good reason it has unlocked the ability
 

5
00:00:05,400 --> 00:00:07,470
good reason it has unlocked the ability
for our computers to perform a whole

6
00:00:07,470 --> 00:00:07,480
for our computers to perform a whole
 

7
00:00:07,480 --> 00:00:10,110
for our computers to perform a whole
host of tasks that we simply didn't know

8
00:00:10,110 --> 00:00:10,120
host of tasks that we simply didn't know
 

9
00:00:10,120 --> 00:00:12,950
host of tasks that we simply didn't know
how to program manually this fascinating
```
produces

```txt
machine learning is a Hot Topic and for
good reason it has unlocked the ability
for our computers to perform a whole
host of tasks that we simply didn't know
how to program manually this fascinating
```

2. Output Format
   - Create a new file with suffix `.clean.txt` (e.g., `How to train simple AIs [EvV5Qtp_fYg].clean.txt`)
   - Maintain the standard naming convention established for video references
