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
