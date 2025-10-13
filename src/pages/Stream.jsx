import { useMemo, useRef, useState, useEffect } from "react";
import { useParams, useLocation, useNavigate } from "react-router-dom";
import {
  Box,
  Container,
  Typography,
  Stack,
  Button,
  Card,
  Chip,
  MenuItem,
  Switch,
  Menu,
  Divider,
  ListItemText,
  Avatar,
  TextField,
  Tabs,
  Tab,
} from "@mui/material";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import Replay10Icon from "@mui/icons-material/Replay10";
import Forward10Icon from "@mui/icons-material/Forward10";
import VolumeUpIcon from "@mui/icons-material/VolumeUp";
import VolumeOffIcon from "@mui/icons-material/VolumeOff";
import PictureInPictureAltIcon from "@mui/icons-material/PictureInPictureAlt";
import SkipNextIcon from "@mui/icons-material/SkipNext";
import MicNoneIcon from "@mui/icons-material/MicNone";
import ClosedCaptionIcon from "@mui/icons-material/ClosedCaption";
import HighQualityIcon from "@mui/icons-material/HighQuality";
import FullscreenIcon from "@mui/icons-material/Fullscreen";
import FullscreenExitIcon from "@mui/icons-material/FullscreenExit";
import FavoriteBorderIcon from "@mui/icons-material/FavoriteBorder";
import AddIcon from "@mui/icons-material/Add";
import useMovies from "../hooks/useMovies";
import Header from "../components/Header";
import { listActors } from "../api/actors";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from "@mui/material";
import { useAuth } from "../hooks/useAuth";
import { listCollections, addMovieToWatchlist } from "../api/watchlist";
import {
  getStreamingInfo,
  startStreaming,
  updateStreamingQuality,
  updatePlaybackPosition,
  stopStreaming,
  getAvailableSubtitles,
} from "../api/streaming";

// Fallback cast data for when a movie lacks actor info
const DEFAULT_STREAM_CAST = [
  { name: "Yoona", avatar: "https://i.pravatar.cc/150?img=47" },
  { name: "Lee Chae-min", avatar: "https://i.pravatar.cc/150?img=12" },
  { name: "Kang Han-na", avatar: "https://i.pravatar.cc/150?img=32" },
  { name: "Choi Gwi-hwa", avatar: "https://i.pravatar.cc/150?img=58" },
  { name: "Oh Eui-sik", avatar: "https://i.pravatar.cc/150?img=15" },
  { name: "Seo Yi-sook", avatar: "https://i.pravatar.cc/150?img=21" },
  { name: "Park Young-woon", avatar: "https://i.pravatar.cc/150?img=13" },
  { name: "Yoon Seo-ah", avatar: "https://i.pravatar.cc/150?img=45" },
];

export default function Stream() {
  const { id } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const { movies } = useMovies();
  const { token, isAuthenticated } = useAuth();

  const movie = useMemo(() => {
    const numericId = Number(id);
    return movies.find(
      (m) => String(m.id) === String(id) || m.id === numericId
    );
  }, [id, movies]);

  const castList = useMemo(() => {
    if (!movie) return DEFAULT_STREAM_CAST;
    if (Array.isArray(movie.actors) && movie.actors.length > 0) {
      return movie.actors.map((a) => {
        if (typeof a === "string") return { name: a };
        const name = a?.name || a?.fullName || a?.username || "";
        const avatar =
          a?.avatar ||
          a?.avatarUrl ||
          a?.imageUrl ||
          a?.image ||
          a?.profilePath;
        const id = a?.id;
        return { id, name, avatar };
      });
    }
    return DEFAULT_STREAM_CAST;
  }, [movie]);

  // Enrich actors by querying API when only names are available
  const [enrichedCast, setEnrichedCast] = useState([]);
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const src = castList.slice(0, 9);
        const out = await Promise.all(
          src.map(async (it) => {
            if ((it?.avatar && it?.id) || !it?.name) return it;
            try {
              const res = await listActors({ q: it.name, page: 0, size: 1 });
              const first = Array.isArray(res?.content)
                ? res.content[0]
                : Array.isArray(res)
                ? res[0]
                : null;
              if (first) {
                return {
                  id: first.id,
                  name: first.name || it.name,
                  avatar:
                    first.imageUrl ||
                    first.image ||
                    first.avatarUrl ||
                    it.avatar,
                };
              }
              return it;
            } catch (_) {
              return it;
            }
          })
        );
        if (mounted) setEnrichedCast(out);
      } catch (_) {
        if (mounted) setEnrichedCast(castList.slice(0, 9));
      }
    })();
    return () => {
      mounted = false;
    };
  }, [castList]);

  // Comments UI state
  const [commentTab, setCommentTab] = useState("comment"); // comment | review
  const [commentText, setCommentText] = useState("");
  const maxCommentLen = 300;

  const episode = location.state?.episode ?? 1;
  const audioMode = location.state?.audioMode ?? "sub";

  const videoRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [volume, setVolume] = useState(1);
  const [muted, setMuted] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [bufferedEnd, setBufferedEnd] = useState(0);
  const [uiVisible, setUiVisible] = useState(true);
  const hideUiTimerRef = useRef(null);
  // Streaming API state
  const [availableQualities, setAvailableQualities] = useState([]);
  const [availableSubtitles, setAvailableSubtitles] = useState([]);
  const [selectedQuality, setSelectedQuality] = useState("1080p");
  const [selectedSubtitleLanguage, setSelectedSubtitleLanguage] =
    useState("English");
  const [subtitleEnabled, setSubtitleEnabled] = useState(true);
  const [videoSrc, setVideoSrc] = useState("");
  const [addOpen, setAddOpen] = useState(false);
  const [collections, setCollections] = useState([]);
  const [addForm, setAddForm] = useState({
    collectionId: "",
    notes: "",
    priority: 1,
  });
  useEffect(() => {
    (async () => {
      try {
        if (!token) return;
        const cols = await listCollections(token);
        setCollections(cols || []);
      } catch (_) {
        // ignore
      }
    })();
  }, [token]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const onLoaded = () => setDuration(v.duration || 0);
    const onTime = () => setCurrentTime(v.currentTime || 0);
    const onProgress = () => {
      try {
        const b = v.buffered;
        if (b && b.length > 0) setBufferedEnd(b.end(b.length - 1));
      } catch (err) {
        // ignore buffered API errors
      }
    };
    v.addEventListener("loadedmetadata", onLoaded);
    v.addEventListener("timeupdate", onTime);
    v.addEventListener("progress", onProgress);
    return () => {
      v.removeEventListener("loadedmetadata", onLoaded);
      v.removeEventListener("timeupdate", onTime);
      v.removeEventListener("progress", onProgress);
    };
  }, []);

  // derive movieId
  const movieId = useMemo(() => {
    const numericId = Number(id);
    return Number.isFinite(numericId) && numericId > 0 ? numericId : movie?.id;
  }, [id, movie]);

  // Bootstrap streaming info and start playback
  useEffect(() => {
    let aborted = false;
    (async () => {
      if (!movieId) return;
      try {
        const info = await getStreamingInfo(movieId);
        if (aborted) return;
        setAvailableQualities(info?.availableQualities || []);
        setAvailableSubtitles(info?.availableSubtitles || []);
        if (info?.currentQuality) setSelectedQuality(info.currentQuality);
        if (info?.currentSubtitleLanguage)
          setSelectedSubtitleLanguage(info.currentSubtitleLanguage);
        if (typeof info?.subtitleEnabled === "boolean")
          setSubtitleEnabled(info.subtitleEnabled);

        // Try enrich subtitles list
        try {
          const subs = await getAvailableSubtitles(movieId);
          if (!aborted && Array.isArray(subs) && subs.length > 0) {
            setAvailableSubtitles(subs);
          }
        } catch (e) {
          // ignore subtitle fetch errors
        }

        const started = await startStreaming({
          movieId,
          quality: info?.currentQuality || selectedQuality,
          subtitleLanguage:
            info?.currentSubtitleLanguage || selectedSubtitleLanguage,
          subtitleEnabled:
            typeof info?.subtitleEnabled === "boolean"
              ? info.subtitleEnabled
              : subtitleEnabled,
          startPosition: info?.currentPositionSeconds || 0,
          autoPlay: true,
        });
        if (aborted) return;
        setAvailableQualities(
          started?.availableQualities || info?.availableQualities || []
        );
        setAvailableSubtitles(
          started?.availableSubtitles || info?.availableSubtitles || []
        );
        setSelectedQuality(
          started?.selectedQuality || info?.currentQuality || selectedQuality
        );
        if (started?.selectedSubtitleLanguage)
          setSelectedSubtitleLanguage(started.selectedSubtitleLanguage);
        setVideoSrc(started?.streamingUrl || "");
      } catch (e) {
        // keep silent; fallback to local movie url if any
      }
    })();
    return () => {
      aborted = true;
    };
  }, [movieId]);

  // Update quality or subtitle options
  const applyQualityOrSubtitleChange = async (next) => {
    if (!movieId) return;
    try {
      const payload = {
        movieId,
        quality: next.quality ?? selectedQuality,
        subtitleLanguage: next.subtitleLanguage ?? selectedSubtitleLanguage,
        subtitleEnabled:
          typeof next.subtitleEnabled === "boolean"
            ? next.subtitleEnabled
            : subtitleEnabled,
        startPosition: Math.floor(currentTime || 0),
        autoPlay: true,
      };
      const res = await updateStreamingQuality(payload);
      setSelectedQuality(res?.selectedQuality ?? payload.quality);
      if (res?.selectedSubtitleLanguage)
        setSelectedSubtitleLanguage(res.selectedSubtitleLanguage);
      if (res?.availableQualities)
        setAvailableQualities(res.availableQualities);
      if (res?.availableSubtitles)
        setAvailableSubtitles(res.availableSubtitles);
      if (res?.streamingUrl) setVideoSrc(res.streamingUrl);
    } catch (e) {
      // ignore
    }
  };

  // Periodically persist playback position while playing
  useEffect(() => {
    if (!movieId || !isPlaying) return;
    const interval = setInterval(() => {
      updatePlaybackPosition({
        movieId,
        currentPosition: Math.floor(currentTime || 0),
        isCompleted: false,
      }).catch(() => {});
    }, 15000);
    return () => clearInterval(interval);
  }, [movieId, isPlaying, currentTime]);

  // Stop streaming on unmount
  useEffect(() => {
    const vAtMount = videoRef.current;
    return () => {
      if (!movieId) return;
      const pos = Math.floor(vAtMount?.currentTime || currentTime || 0);
      const dur = Math.floor(vAtMount?.duration || duration || 0);
      const completed = dur > 0 && pos >= Math.max(0, dur - 3);
      stopStreaming({
        movieId,
        currentPosition: pos,
        isCompleted: completed,
      }).catch(() => {});
    };
  }, [movieId, currentTime, duration]);

  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) {
      v.play();
      setIsPlaying(true);
      showUiTemporarily();
    } else {
      v.pause();
      setIsPlaying(false);
      showUi();
    }
  };
  const seekBy = (sec) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = Math.max(0, Math.min((v.currentTime || 0) + sec, duration));
  };
  const onScrub = (e) => {
    const v = videoRef.current;
    if (!v) return;
    const val = Number(e.target.value);
    v.currentTime = val;
    setCurrentTime(val);
  };
  const toggleMute = () => {
    const v = videoRef.current;
    if (!v) return;
    v.muted = !muted;
    setMuted(v.muted);
  };
  const onVolume = (e) => {
    const v = videoRef.current;
    if (!v) return;
    const val = Number(e.target.value);
    v.volume = val;
    setVolume(val);
    if (val > 0 && muted) {
      v.muted = false;
      setMuted(false);
    }
  };
  const toggleFullscreen = () => {
    const container = document.getElementById("player-container");
    if (!container) return;
    if (!document.fullscreenElement) {
      container.requestFullscreen?.();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen?.();
      setIsFullscreen(false);
    }
  };
  const enterPiP = async () => {
    try {
      const v = videoRef.current;
      if (v && document.pictureInPictureEnabled && !v.disablePictureInPicture) {
        // @ts-ignore browser API
        await v.requestPictureInPicture();
      }
    } catch (err) {
      // ignore PiP errors
    }
  };

  const fmt = (s) => {
    const t = Math.floor(s || 0);
    const h = Math.floor(t / 3600);
    const m = Math.floor((t % 3600) / 60);
    const sec = t % 60;
    const mm = h > 0 ? String(m).padStart(2, "0") : String(m);
    const ss = String(sec).padStart(2, "0");
    return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
  };

  // UI visibility helpers
  const clearHideUiTimer = () => {
    if (hideUiTimerRef.current) {
      clearTimeout(hideUiTimerRef.current);
      hideUiTimerRef.current = null;
    }
  };
  const showUi = () => {
    setUiVisible(true);
    clearHideUiTimer();
  };
  const scheduleHideUi = () => {
    clearHideUiTimer();
    if (!isPlaying) return;
    hideUiTimerRef.current = setTimeout(() => setUiVisible(false), 2000);
  };
  const showUiTemporarily = () => {
    setUiVisible(true);
    scheduleHideUi();
  };
  const onMouseMovePlayer = () => {
    showUiTemporarily();
  };

  // Menus anchors
  const [anchorQuality, setAnchorQuality] = useState(null);
  const [anchorSubtitle, setAnchorSubtitle] = useState(null);

  // Helpers
  const formatBytes = (bytes) => {
    if (!bytes || bytes <= 0) return "-";
    const units = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    const val = bytes / Math.pow(1024, i);
    return `${val.toFixed(val >= 100 ? 0 : val >= 10 ? 1 : 2)} ${units[i]}`;
  };

  useEffect(() => {
    return () => clearHideUiTimer();
  }, []);

  return (
    <Box sx={{ minHeight: "100vh", color: "#fff", background: "#0a0f1a" }}>
      <Header />
      <Container maxWidth="xl" sx={{ py: 4, mt: { xs: 6, md: 8 }, px: 3 }}>
        <Button
          onClick={() => navigate(-1)}
          startIcon={<ArrowBackIcon />}
          sx={{ color: "#FFD700", mb: 2 }}
        >
          Quay lại
        </Button>

        <Typography variant="h5" fontWeight={900} sx={{ mb: 2 }}>
          {movie?.title || "Đang phát"}
        </Typography>
        {false && (
          <Typography
            variant="body2"
            sx={{ color: "rgba(255,255,255,0.75)", mb: 3 }}
          >
            Tập {episode} • {audioMode === "dub" ? "Thuyết minh" : "Phụ đề"}
          </Typography>
        )}

        <Card sx={{ background: "#000", borderRadius: 2, overflow: "hidden" }}>
          <Box
            id="player-container"
            onMouseMove={onMouseMovePlayer}
            sx={{
              position: "relative",
              background: "#000",
              cursor: uiVisible ? "auto" : "none",
              width: "100%",
              maxWidth: { xs: "100%" },
              mx: "auto",
            }}
          >
            {/* Aspect ratio wrapper to prevent any layout jump */}
            <Box sx={{ position: "relative", width: "100%", pt: "56.25%" }}>
              <style>{`
              video::cue { color: #fff; font-weight: 700; text-shadow: 0 0 2px #000, 0 0 4px #000, 0 0 6px #000; }
              .player-slider, .volume-slider { -webkit-appearance: none; appearance: none; background: transparent; height: 6px; cursor: pointer; }
              .player-slider::-webkit-slider-runnable-track, .volume-slider::-webkit-slider-runnable-track { height: 6px; background: transparent; }
              .player-slider::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 14px; height: 14px; border-radius: 50%; background: #ffffff; border: 2px solid rgba(0,0,0,0.9); margin-top: -4px; box-shadow: 0 2px 8px rgba(0,0,0,0.35); }
              .volume-slider::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 12px; height: 12px; border-radius: 50%; background: #ffffff; border: 2px solid rgba(0,0,0,0.9); ; box-shadow: 0 2px 8px rgba(0,0,0,0.35); }
              .player-slider:focus-visible::-webkit-slider-thumb, .volume-slider:focus-visible::-webkit-slider-thumb { outline: none; box-shadow: 0 0 0 6px rgba(255,255,255,0.15); }
              .player-slider::-moz-range-track, .volume-slider::-moz-range-track { height: 6px; background: transparent; }
              .player-slider::-moz-range-thumb { width: 14px; height: 14px; border-radius: 50%; background: #ffffff; border: 2px solid rgba(0,0,0,0.9); box-shadow: 0 2px 8px rgba(0,0,0,0.35); }
              .volume-slider::-moz-range-thumb { width: 12px; height: 12px; border-radius: 50%; background: #ffffff; border: 2px solid rgba(0,0,0,0.9); box-shadow: 0 2px 8px rgba(0,0,0,0.35); }
              `}</style>
              <video
                ref={videoRef}
                src={
                  (videoSrc || movie?.streamingUrl || movie?.videoUrl) ??
                  undefined
                }
                poster={movie?.backdrop || movie?.thumb}
                style={{
                  position: "absolute",
                  inset: 0,
                  width: "100%",
                  height: "100%",
                  display: "block",
                  objectFit: "contain",
                  background: "#000",
                }}
                preload="metadata"
                onClick={togglePlay}
                onPlay={() => {
                  setIsPlaying(true);
                  showUiTemporarily();
                }}
                onPause={() => {
                  setIsPlaying(false);
                  showUi();
                  if (movieId) {
                    updatePlaybackPosition({
                      movieId,
                      currentPosition: Math.floor(
                        videoRef.current?.currentTime || 0
                      ),
                      isCompleted: false,
                    }).catch(() => {});
                  }
                }}
              />
            </Box>

            {/* Progress */}
            <Box
              sx={{
                position: "absolute",
                left: 0,
                right: 0,
                bottom: 64,
                px: 2,
                opacity: uiVisible ? 1 : 0,
                transform: uiVisible ? "translateY(0)" : "translateY(6px)",
                transition: "opacity .2s ease, transform .2s ease",
                pointerEvents: uiVisible ? "auto" : "none",
              }}
            >
              <Box sx={{ position: "relative", height: 6 }}>
                <Box
                  sx={{
                    position: "absolute",
                    inset: 0,
                    bgcolor: "rgba(255,255,255,0.12)",
                    borderRadius: 999,
                    boxShadow: "inset 0 1px 1px rgba(0,0,0,0.6)",
                  }}
                />
                <Box
                  sx={{
                    position: "absolute",
                    left: 0,
                    top: 0,
                    bottom: 0,
                    width: `${Math.min(
                      100,
                      (bufferedEnd / Math.max(1, duration)) * 100
                    )}%`,
                    bgcolor: "rgba(255,255,255,0.28)",
                    borderRadius: 999,
                  }}
                />
                <Box
                  sx={{
                    position: "absolute",
                    left: 0,
                    top: 0,
                    bottom: 0,
                    width: `${Math.min(
                      100,
                      (currentTime / Math.max(1, duration)) * 100
                    )}%`,
                    background: "linear-gradient(90deg, #fff 0%, #e6f0ff 100%)",
                    borderRadius: 999,
                    boxShadow: "0 0 10px rgba(255,255,255,0.25)",
                  }}
                />
                <input
                  type="range"
                  min={0}
                  max={Math.max(1, duration)}
                  step={0.1}
                  value={currentTime}
                  onChange={onScrub}
                  className="player-slider"
                  style={{
                    position: "absolute",
                    inset: 0,
                    width: "100%",
                    background: "transparent",
                    outline: "none",
                  }}
                />
              </Box>
            </Box>

            {/* Controls */}
            <Stack
              direction="row"
              alignItems="center"
              justifyContent="space-between"
              sx={{
                position: "absolute",
                left: 0,
                right: 0,
                bottom: 0,
                px: 2.5,
                py: 1.25,
                background:
                  "linear-gradient(180deg, rgba(0,0,0,0.00) 0%, rgba(0,0,0,0.55) 35%, rgba(0,0,0,0.75) 100%)",
                backdropFilter: "blur(8px)",
                borderTop: "1px solid rgba(255,255,255,0.08)",
                opacity: uiVisible ? 1 : 0,
                transform: uiVisible ? "translateY(0)" : "translateY(8px)",
                transition: "opacity .2s ease, transform .2s ease",
                pointerEvents: uiVisible ? "auto" : "none",
              }}
            >
              <Stack direction="row" spacing={1.25} alignItems="center">
                <Button
                  onClick={togglePlay}
                  sx={{
                    minWidth: 0,
                    color: "#fff",
                    p: 0.75,
                    borderRadius: 999,
                    transition: "all .2s",
                    "&:hover": { bgcolor: "rgba(255,255,255,0.08)" },
                  }}
                >
                  {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
                </Button>
                <Button
                  onClick={() => seekBy(-10)}
                  sx={{
                    minWidth: 0,
                    color: "#fff",
                    p: 0.75,
                    borderRadius: 999,
                    transition: "all .2s",
                    "&:hover": { bgcolor: "rgba(255,255,255,0.08)" },
                  }}
                >
                  <Replay10Icon />
                </Button>
                <Button
                  onClick={() => seekBy(10)}
                  sx={{
                    minWidth: 0,
                    color: "#fff",
                    p: 0.75,
                    borderRadius: 999,
                    transition: "all .2s",
                    "&:hover": { bgcolor: "rgba(255,255,255,0.08)" },
                  }}
                >
                  <Forward10Icon />
                </Button>
                <Typography
                  variant="caption"
                  sx={{ color: "#fff", fontWeight: 700 }}
                >
                  {fmt(currentTime)} / {fmt(duration)}
                </Typography>
                <Stack
                  direction="row"
                  spacing={1}
                  alignItems="center"
                  sx={{ ml: 1 }}
                >
                  <Button
                    onClick={toggleMute}
                    sx={{
                      minWidth: 0,
                      color: "#fff",
                      p: 0.75,
                      borderRadius: 999,
                      transition: "all .2s",
                      "&:hover": { bgcolor: "rgba(255,255,255,0.08)" },
                    }}
                  >
                    {muted || volume === 0 ? (
                      <VolumeOffIcon />
                    ) : (
                      <VolumeUpIcon />
                    )}
                  </Button>
                  <Box
                    sx={{
                      position: "relative",
                      width: { xs: 110, sm: 140, md: 160 },
                      height: 6,
                    }}
                  >
                    <Box
                      sx={{
                        position: "absolute",
                        inset: 0,
                        bgcolor: "rgba(255,255,255,0.18)",
                        borderRadius: 999,
                        boxShadow: "inset 0 1px 1px rgba(0,0,0,0.6)",
                      }}
                    />
                    <Box
                      sx={{
                        position: "absolute",
                        left: 0,
                        top: 0,
                        bottom: 0,
                        width: `${(muted ? 0 : volume) * 100}%`,
                        background:
                          "linear-gradient(90deg, #fff 0%, #e6f0ff 100%)",
                        borderRadius: 999,
                        boxShadow: "0 0 10px rgba(255,255,255,0.2)",
                      }}
                    />
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={muted ? 0 : volume}
                      onChange={onVolume}
                      className="volume-slider"
                      style={{
                        position: "absolute",
                        inset: -3,
                        width: "calc(100% + 8px)",
                        background: "transparent",
                        outline: "none",
                      }}
                    />
                  </Box>
                </Stack>
              </Stack>
              <Stack direction="row" spacing={1.0} alignItems="center">
                <Button
                  sx={{
                    minWidth: 0,
                    color: "#fff",
                    p: 0.75,
                    borderRadius: 999,
                    "&:hover": { bgcolor: "rgba(255,255,255,0.08)" },
                  }}
                >
                  <SkipNextIcon />
                </Button>
                <Button
                  sx={{
                    minWidth: 0,
                    color: "#fff",
                    p: 0.75,
                    borderRadius: 999,
                    "&:hover": { bgcolor: "rgba(255,255,255,0.08)" },
                  }}
                >
                  <MicNoneIcon />
                </Button>
                {/* Subtitles menu trigger */}
                <Button
                  onClick={(e) => setAnchorSubtitle(e.currentTarget)}
                  sx={{
                    minWidth: 0,
                    color: "#fff",
                    p: 0.75,
                    borderRadius: 999,
                    "&:hover": { bgcolor: "rgba(255,255,255,0.08)" },
                  }}
                >
                  <ClosedCaptionIcon />
                </Button>
                <Button
                  onClick={enterPiP}
                  sx={{
                    minWidth: 0,
                    color: "#fff",
                    p: 0.75,
                    borderRadius: 999,
                    "&:hover": { bgcolor: "rgba(255,255,255,0.08)" },
                  }}
                >
                  <PictureInPictureAltIcon />
                </Button>
                {/* Quality menu trigger */}
                <Button
                  onClick={(e) => setAnchorQuality(e.currentTarget)}
                  sx={{
                    minWidth: 0,
                    color: "#fff",
                    p: 0.75,
                    borderRadius: 999,
                    "&:hover": { bgcolor: "rgba(255,255,255,0.08)" },
                  }}
                >
                  <HighQualityIcon />
                </Button>
                <Button
                  onClick={toggleFullscreen}
                  sx={{
                    minWidth: 0,
                    color: "#fff",
                    p: 0.75,
                    borderRadius: 999,
                    "&:hover": { bgcolor: "rgba(255,255,255,0.08)" },
                  }}
                >
                  {isFullscreen ? <FullscreenExitIcon /> : <FullscreenIcon />}
                </Button>
              </Stack>
            </Stack>
            {/* Quality Menu */}
            <Menu
              anchorEl={anchorQuality}
              open={Boolean(anchorQuality)}
              onClose={() => setAnchorQuality(null)}
              PaperProps={{ sx: { bgcolor: "#0f1422", color: "#fff" } }}
            >
              {(availableQualities?.length
                ? availableQualities
                : [selectedQuality]
              ).map((q) => (
                <MenuItem
                  key={q}
                  selected={q === selectedQuality}
                  onClick={() => {
                    setAnchorQuality(null);
                    applyQualityOrSubtitleChange({ quality: q });
                  }}
                >
                  <ListItemText>{q}</ListItemText>
                </MenuItem>
              ))}
            </Menu>
            {/* Subtitles Menu */}
            <Menu
              anchorEl={anchorSubtitle}
              open={Boolean(anchorSubtitle)}
              onClose={() => setAnchorSubtitle(null)}
              PaperProps={{
                sx: { bgcolor: "#0f1422", color: "#fff", minWidth: 220 },
              }}
            >
              <MenuItem disableRipple disableTouchRipple>
                <Stack direction="row" alignItems="center" spacing={1}>
                  <Typography variant="body2">Phụ đề</Typography>
                  <Switch
                    size="small"
                    checked={subtitleEnabled}
                    onChange={(e) => {
                      setSubtitleEnabled(e.target.checked);
                      applyQualityOrSubtitleChange({
                        subtitleEnabled: e.target.checked,
                      });
                    }}
                  />
                </Stack>
              </MenuItem>
              <Divider sx={{ borderColor: "rgba(255,255,255,0.12)" }} />
              {(availableSubtitles?.length ? availableSubtitles : []).map(
                (s) => (
                  <MenuItem
                    key={s.languageCode || s.language}
                    selected={s.language === selectedSubtitleLanguage}
                    onClick={() => {
                      setAnchorSubtitle(null);
                      applyQualityOrSubtitleChange({
                        subtitleLanguage: s.language,
                      });
                    }}
                  >
                    <ListItemText>{s.language}</ListItemText>
                  </MenuItem>
                )
              )}
            </Menu>
          </Box>
        </Card>

        {/* Metadata bar below player */}
        <Box
          sx={{
            mt: 1.5,
            px: 1.5,
            py: 1,
            display: "flex",
            alignItems: "center",
            flexWrap: "wrap",
            rowGap: 1,
            columnGap: 1,
            background:
              "linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.02) 100%)",
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: 2,
            boxShadow: "0 6px 18px rgba(0,0,0,0.25)",
          }}
        >
          {selectedQuality && (
            <Chip
              size="small"
              label={selectedQuality.toUpperCase()}
              sx={{ bgcolor: "rgba(255,255,255,0.08)", color: "#fff" }}
            />
          )}
          {movie?.year && (
            <Chip
              size="small"
              label={movie.year}
              sx={{ bgcolor: "rgba(255,255,255,0.08)", color: "#fff" }}
            />
          )}
          {(movie?.videoDuration || movie?.duration) && (
            <Chip
              size="small"
              label={`${movie?.videoDuration ?? movie?.duration} phút`}
              sx={{ bgcolor: "rgba(255,255,255,0.08)", color: "#fff" }}
            />
          )}
          {movie?.videoFormat && (
            <Chip
              size="small"
              label={movie.videoFormat.toUpperCase()}
              sx={{ bgcolor: "rgba(255,255,255,0.08)", color: "#fff" }}
            />
          )}
          {movie?.fileSizeBytes && (
            <Chip
              size="small"
              label={formatBytes(movie.fileSizeBytes)}
              sx={{ bgcolor: "rgba(255,255,255,0.08)", color: "#fff" }}
            />
          )}
          {movie?.language && (
            <Chip
              size="small"
              label={movie.language}
              sx={{ bgcolor: "rgba(255,255,255,0.08)", color: "#fff" }}
            />
          )}
          <Box sx={{ flex: 1 }} />
          {movie?.trailerUrl && (
            <Button
              component="a"
              href={movie.trailerUrl}
              target="_blank"
              rel="noreferrer"
              sx={{ color: "#fff" }}
            >
              Trailer
            </Button>
          )}
          <Button startIcon={<FavoriteBorderIcon />} sx={{ color: "#fff" }}>
            Yêu thích
          </Button>
          <Button
            startIcon={<AddIcon />}
            sx={{ color: "#fff" }}
            onClick={() => {
              if (!isAuthenticated) {
                navigate("/login");
                return;
              }
              setAddOpen(true);
            }}
          >
            Thêm vào
          </Button>
        </Box>
        <Divider sx={{ my: 3, borderColor: "rgba(255,255,255,0.08)" }} />

        {/* Info + Versions + Comments (left) and Cast (right) */}
        <Box>
          <Stack
            direction={{ xs: "column", md: "row" }}
            spacing={3}
            alignItems="flex-start"
          >
            {/* Left column */}
            <Box sx={{ flex: 1.5, minWidth: 300 }}>
              {/* Header-like summary under player: poster + title + chips + genres + overview link */}
              <Box sx={{}}>
                <Stack
                  direction={{ xs: "column", md: "row" }}
                  spacing={3}
                  alignItems="flex-start"
                >
                  {/* Poster */}
                  <Box sx={{ width: { xs: 140, sm: 160 }, flexShrink: 0 }}>
                    <Box
                      component="img"
                      src={movie?.thumb}
                      alt={movie?.title}
                      sx={{
                        width: "100%",
                        aspectRatio: "2 / 3",
                        objectFit: "cover",
                        borderRadius: 2,
                        border: "1px solid rgba(255,255,255,0.1)",
                      }}
                    />
                  </Box>
                  {/* Text block */}
                  <Box sx={{ flex: 1 }}>
                    <Typography variant="h6" fontWeight={900} sx={{ mb: 0.5 }}>
                      {movie?.title}
                    </Typography>
                    {movie?.englishTitle && (
                      <Typography
                        variant="subtitle2"
                        sx={{ color: "rgba(255,255,255,0.75)", mb: 1 }}
                      >
                        {movie.englishTitle}
                      </Typography>
                    )}
                    {/* Chips row: quality, age, year, runtime */}
                    <Stack
                      direction="row"
                      alignItems="center"
                      spacing={1}
                      sx={{ mb: 1.25, flexWrap: "wrap" }}
                    >
                      {selectedQuality && (
                        <Chip
                          size="small"
                          label={selectedQuality.toUpperCase()}
                          sx={{
                            bgcolor: "rgba(255,255,255,0.1)",
                            color: "#fff",
                          }}
                        />
                      )}
                      {movie?.ageRating && (
                        <Chip
                          size="small"
                          label={movie.ageRating}
                          sx={{
                            bgcolor: "rgba(255,255,255,0.1)",
                            color: "#fff",
                          }}
                        />
                      )}
                      {movie?.year && (
                        <Chip
                          size="small"
                          label={movie.year}
                          sx={{
                            bgcolor: "rgba(255,255,255,0.1)",
                            color: "#fff",
                          }}
                        />
                      )}
                      {(movie?.videoDuration || movie?.duration) && (
                        <Chip
                          size="small"
                          label={`${movie?.videoDuration ?? movie?.duration}`}
                          sx={{
                            bgcolor: "rgba(255,255,255,0.1)",
                            color: "#fff",
                          }}
                        />
                      )}
                    </Stack>
                    {/* Genre tags */}
                    {Array.isArray(movie?.genres) &&
                      movie.genres.length > 0 && (
                        <Stack
                          direction="row"
                          spacing={1}
                          alignItems="center"
                          sx={{ mb: 1.25, flexWrap: "wrap", rowGap: 1 }}
                        >
                          {movie.genres.map((g, i) => (
                            <Chip
                              key={`${g}-${i}`}
                              size="small"
                              label={g}
                              sx={{
                                bgcolor: "rgba(255,255,255,0.08)",
                                color: "#fff",
                              }}
                            />
                          ))}
                        </Stack>
                      )}
                  </Box>
                  {/* Overview snippet with link */}
                  <Box sx={{ flex: 1 }}>
                    {movie?.synopsis && (
                      <Typography
                        variant="body2"
                        sx={{ color: "rgba(255,255,255,0.85)", mb: 1.25 }}
                      >
                        {movie.synopsis}
                      </Typography>
                    )}
                    <Button
                      onClick={() => navigate(`/movie/${movie?.id}`)}
                      sx={{
                        color: "#FFD700",
                        px: 0,
                        textTransform: "none",
                        fontWeight: 700,
                      }}
                    >
                      Thông tin phim ›
                    </Button>
                  </Box>
                </Stack>
              </Box>
              <Divider sx={{ my: 2, borderColor: "rgba(255,255,255,0.08)" }} />

              {/* Available Versions */}
              <Typography variant="h6" fontWeight={900} sx={{ mb: 3, mt: 3 }}>
                Các bản chiếu
              </Typography>
              <Card
                sx={{
                  background: "#0f1013",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 2,
                  overflow: "hidden",
                  mb: 3,
                }}
              >
                <Stack direction="row" alignItems="stretch">
                  <Box sx={{ flex: 1, p: 2.5 }}>
                    <Stack
                      direction="row"
                      alignItems="center"
                      spacing={1.25}
                      sx={{ mb: 1.25 }}
                    >
                      <ClosedCaptionIcon
                        sx={{ fontSize: 18, color: "#9fb3c8" }}
                      />
                      <Typography
                        variant="body2"
                        sx={{ color: "#9fb3c8", fontWeight: 700 }}
                      >
                        Phụ đề
                      </Typography>
                    </Stack>
                    <Typography
                      variant="subtitle1"
                      fontWeight={800}
                      sx={{ mb: 1, pr: 2 }}
                    >
                      {movie?.title}
                    </Typography>
                    <Stack
                      direction="row"
                      spacing={1}
                      alignItems="center"
                      sx={{ mb: 2, flexWrap: "wrap" }}
                    >
                      {movie?.videoQuality && (
                        <Chip
                          size="small"
                          label={movie.videoQuality.toUpperCase()}
                          sx={{
                            bgcolor: "rgba(255,255,255,0.08)",
                            color: "#fff",
                          }}
                        />
                      )}
                      {movie?.year && (
                        <Chip
                          size="small"
                          label={movie.year}
                          sx={{
                            bgcolor: "rgba(255,255,255,0.08)",
                            color: "#fff",
                          }}
                        />
                      )}
                      {movie?.videoDuration && (
                        <Chip
                          size="small"
                          label={`${movie.videoDuration} phút`}
                          sx={{
                            bgcolor: "rgba(255,255,255,0.08)",
                            color: "#fff",
                          }}
                        />
                      )}
                    </Stack>
                    <Button
                      variant="outlined"
                      onClick={() => navigate(`/stream/${movie?.id}`)}
                      sx={{
                        color: "#fff",
                        borderColor: "rgba(255,255,255,0.2)",
                      }}
                    >
                      Đang xem
                    </Button>
                  </Box>
                  <Box
                    sx={{ width: { xs: 140, sm: 220 }, position: "relative" }}
                  >
                    <Box
                      component="img"
                      src={movie?.thumb}
                      alt={movie?.title}
                      sx={{ height: "100%", width: "100%", objectFit: "cover" }}
                    />
                  </Box>
                </Stack>
              </Card>
              <Divider sx={{ my: 2, borderColor: "rgba(255,255,255,0.08)" }} />

              {/* Comments & Reviews */}
              <Typography variant="h6" fontWeight={900} sx={{ mb: 1 }}>
                Bình luận & Đánh giá
              </Typography>
              <Tabs
                value={commentTab}
                onChange={(_, v) => setCommentTab(v)}
                textColor="primary"
                TabIndicatorProps={{ style: { backgroundColor: "#FFD700" } }}
                sx={{
                  minHeight: 36,
                  mb: 1,
                  "& .MuiTab-root": { minHeight: 36, color: "#fff" },
                }}
              >
                <Tab label="Bình luận" value="comment" />
                <Tab label="Đánh giá" value="review" />
              </Tabs>
              <Stack
                direction="row"
                spacing={2}
                alignItems="flex-start"
                sx={{
                  p: 2,
                  bgcolor: "rgba(15,16,19,0.9)",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 2,
                  boxShadow: "0 10px 30px rgba(0,0,0,0.25)",
                  mb: 2,
                }}
              >
                <Avatar alt="Bạn" src="" sx={{ width: 36, height: 36 }} />
                <Box sx={{ flex: 1 }}>
                  <TextField
                    placeholder={
                      commentTab === "comment"
                        ? "Viết bình luận..."
                        : "Viết đánh giá..."
                    }
                    multiline
                    minRows={2}
                    variant="outlined"
                    fullWidth
                    value={commentText}
                    onChange={(e) =>
                      setCommentText(e.target.value.slice(0, maxCommentLen))
                    }
                    helperText={`${commentText.length}/${maxCommentLen}`}
                    InputProps={{ sx: { color: "#fff" } }}
                    sx={{
                      "& .MuiOutlinedInput-root": {
                        color: "#fff",
                        bgcolor: "#111",
                        "& fieldset": { borderColor: "rgba(255,255,255,0.2)" },
                        "&:hover fieldset": {
                          borderColor: "rgba(255,255,255,0.35)",
                        },
                        "&.Mui-focused fieldset": { borderColor: "#FFD700" },
                      },
                      "& .MuiFormHelperText-root": {
                        color: "rgba(255,255,255,0.6)",
                      },
                    }}
                  />
                  <Stack
                    direction="row"
                    justifyContent="flex-end"
                    sx={{ mt: 1 }}
                  >
                    <Button
                      sx={{
                        color: "#000",
                        bgcolor: "#FFD700",
                        fontWeight: 700,
                        px: 2,
                        borderRadius: 999,
                        ":hover": { bgcolor: "#FFC107" },
                      }}
                    >
                      Gửi
                    </Button>
                  </Stack>
                </Box>
              </Stack>

              {/* Sample comments list */}
              <Stack spacing={2}>
                {[1, 2, 3].map((i) => (
                  <Stack key={i} direction="row" spacing={2}>
                    <Avatar sx={{ width: 36, height: 36 }}>U{i}</Avatar>
                    <Box sx={{ flex: 1 }}>
                      <Stack direction="row" alignItems="center" spacing={1}>
                        <Typography variant="subtitle2" fontWeight={700}>
                          Người dùng {i}
                        </Typography>
                        <Typography
                          variant="caption"
                          color="rgba(255,255,255,0.6)"
                        >
                          11 phút trước
                        </Typography>
                      </Stack>
                      <Typography
                        variant="body2"
                        sx={{ color: "rgba(255,255,255,0.85)" }}
                      >
                        Bình luận mẫu để minh hoạ giao diện.
                      </Typography>
                    </Box>
                  </Stack>
                ))}
              </Stack>
            </Box>

            <Divider
              orientation="vertical"
              flexItem
              sx={{
                display: { xs: "none", md: "block" },
                borderColor: "rgba(255,255,255,0.08)",
                mx: 0.5,
              }}
            />

            {/* Right sidebar: Cast */}
            <Box sx={{ width: { xs: "100%", md: 320 } }}>
              <Typography variant="h6" fontWeight={900} sx={{ mb: 1.5 }}>
                Diễn viên
              </Typography>
              <Box
                sx={{
                  display: "grid",
                  gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
                  gap: 2,
                }}
              >
                {(enrichedCast.length > 0 ? enrichedCast : castList)
                  .slice(0, 9)
                  .map((c, idx) => {
                    const item = typeof c === "string" ? { name: c } : c;
                    const fallbackAvatar = `https://i.pravatar.cc/150?img=${
                      (idx % 70) + 1
                    }`;
                    return (
                      <Stack
                        key={`${item.name ?? idx}-${idx}`}
                        alignItems="center"
                        spacing={1}
                        onClick={() => {
                          if (item?.id) navigate(`/actor/${item.id}`);
                          else if (item?.name)
                            navigate(
                              `/actors?q=${encodeURIComponent(item.name)}`
                            );
                        }}
                        sx={{ cursor: "pointer" }}
                      >
                        <Avatar
                          src={item.avatar || fallbackAvatar}
                          alt={item.name}
                          sx={{ width: 70, height: 70 }}
                          imgProps={{
                            onError: (e) => {
                              e.currentTarget.src = fallbackAvatar;
                            },
                          }}
                        >
                          {(item.name || "?").charAt(0)}
                        </Avatar>
                        <Typography
                          variant="caption"
                          noWrap
                          sx={{
                            maxWidth: 140,
                            textAlign: "center",
                            fontSize: 14,
                          }}
                        >
                          {item.name || "Diễn viên"}
                        </Typography>
                      </Stack>
                    );
                  })}
              </Box>
            </Box>
          </Stack>
        </Box>
      </Container>
      <Dialog
        open={addOpen}
        onClose={() => setAddOpen(false)}
        fullWidth
        maxWidth="sm"
      >
        <DialogTitle>Thêm vào danh sách</DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ mt: 1 }}>
            <TextField
              select
              label="Bộ sưu tập"
              value={addForm.collectionId}
              onChange={(e) =>
                setAddForm((p) => ({ ...p, collectionId: e.target.value }))
              }
            >
              {collections.map((c) => (
                <MenuItem key={c.id} value={c.id}>
                  {c.name}
                </MenuItem>
              ))}
            </TextField>
            <TextField
              label="Ghi chú"
              value={addForm.notes}
              onChange={(e) =>
                setAddForm((p) => ({ ...p, notes: e.target.value }))
              }
            />
            <TextField
              label="Ưu tiên"
              type="number"
              value={addForm.priority}
              onChange={(e) =>
                setAddForm((p) => ({
                  ...p,
                  priority: Number(e.target.value || 0),
                }))
              }
            />
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddOpen(false)}>Hủy</Button>
          <Button
            variant="contained"
            onClick={async () => {
              try {
                if (!movieId) return;
                await addMovieToWatchlist(movieId, addForm, token);
                setAddOpen(false);
              } catch (_) {
                // ignore
              }
            }}
          >
            Thêm
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
