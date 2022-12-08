# 💃💃 in sync
Your personal AI synchronization assistant.
<br>

https://user-images.githubusercontent.com/113004083/206076584-8894fda5-c629-41f0-88f4-abdf5d016330.mp4

_Dance video sourced from Urban Dance Camp YouTube channel._

## Application Frontend
Project explained in further detail at https://github.com/xkeeja/insync-frontend.

## Getting Started
### Setup

Navigate to the base level of the repository
```
cd {your/path/here}/insync-backend
```

Install package, requirements, & dependencies
```
pip install -U pip
pip install .
```

Install ffmpeg for backend video compression

_on linux_
```
sudo apt update
sudo apt install ffmpeg
```
_on macOS_
```
brew install ffmpeg
```
_for other systems or direct download_

https://ffmpeg.org/download.html



### ENV Variables
Create `.env` file
```
touch .env
```
Inside `.env`, set the following variables
```
BUCKET=your_own_google_bucket_name
```

### Local API testing
```
make run_api
```

### Docker image for deployment
```
docker build -t {your_details_here} .
```

## Built With
- [Python](https://www.python.org/) - Frontend & Backend
- [Streamlit](https://streamlit.io/) - Frontend Deployment
- [GCP](https://cloud.google.com/) - Storage & Backend Deployment
- [TensorFlow](https://tfhub.dev/google/movenet/multipose/lightning/1) - Pose Detection Model

## Acknowledgements
Inspired by [Kanami](https://www.linkedin.com/in/kanami-oyama-9a666b243/)'s love of dance.

## Team Members
- Kanami Oyama ([GitHub](https://github.com/kanpinpon)) ([LinkedIn](https://www.linkedin.com/in/kanami-oyama-9a666b243/))
- Jaylon Saville ([GitHub](https://github.com/jaysaville)) ([LinkedIn](https://www.linkedin.com/in/jaysaville/))
- Vincent-Victor Rodriguez--Le Roy ([GitHub](https://github.com/Slokem)) ([LinkedIn](https://www.linkedin.com/in/vincent-victor-r-328aa5a8/))

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License
