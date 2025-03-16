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

After downloading the raw transcripts, we follow these comprehensive steps to create cleaned, readable versions:

#### 1. Deduplication and Content Extraction

- Remove timestamp information (except for key section markers)
- Consolidate fragmented subtitle text into complete sentences
- Remove duplicated phrases from auto-generation
- Eliminate filler content and unnecessary repetition

**Before:**
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
```

**After:**
```
Machine learning is a hot topic and for good reason. It has unlocked the ability
for our computers to perform a whole host of tasks that we simply didn't know 
how to program manually.
```

#### 2. Content Organization and Structure

- Add a structured metadata header with:
  - Title and video ID
  - Cleanup notes explaining transformations made
- Create clear section headers using markdown (e.g., # INTRODUCTION)
- Group related topics into logical sections
- Preserve key timestamps for reference points (e.g., [00:10:53])
- Add context for visual demonstrations

#### 3. Formatting and Readability

- Use consistent line wrapping at approximately 80 characters
- Apply proper paragraph breaks with appropriate spacing
- Use markdown formatting for headers and structure
- Standardize technical terminology (e.g., "DAGs - Directed Acyclic Graphs")
- Maintain consistent capitalization and punctuation

#### 4. Output Format

- Create a new file with suffix `.clean.srt` (e.g., `How to train simple AIs [EvV5Qtp_fYg].clean.srt`)
- Maintain the standard naming convention established for video references
- Include the following standard sections:
  - Introduction
  - Core technical content (varies by video)
  - Results/demonstrations
  - Conclusion

#### Example Metadata Header

```
---
Title: How to train simple AIs
VideoID: EvV5Qtp_fYg
Cleanup Notes:
- Removed timestamp information except for key sections
- Consolidated subtitle fragments into complete sentences
- Removed duplicated text from auto-generation
- Added section headers for clarity
- Preserved context for visual demonstrations
---
```

This comprehensive cleanup process ensures our reference materials are clear, concise, and maximally useful for implementation while maintaining traceability to source content.
