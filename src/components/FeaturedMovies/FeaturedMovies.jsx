import { memo, useMemo, useRef } from "react";
import { useNavigate } from "react-router-dom";
import PropTypes from "prop-types";
import {
  Box,
  Container,
  Typography,
  Card,
  CardMedia,
  Chip,
  IconButton,
  Tooltip,
  keyframes,
} from "@mui/material";
import { styled } from "@mui/material/styles";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import FavoriteBorderIcon from "@mui/icons-material/FavoriteBorder";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import StarIcon from "@mui/icons-material/Star";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import LocalMoviesIcon from "@mui/icons-material/LocalMovies";
import LocalOfferIcon from "@mui/icons-material/LocalOffer";
import { addFavorite as addFavoriteApi } from "../../api/favorites";

// Animations
const slideIn = keyframes`
  from { opacity: 0; transform: translateX(30px); }
  to { opacity: 1; transform: translateX(0); }
`;

const glowEffect = keyframes`
  0%, 100% { box-shadow: 0 0 20px rgba(255, 215, 0, 0.3); }
  50% { box-shadow: 0 0 30px rgba(255, 215, 0, 0.6); }
`;

const SectionContainer = styled(Box)(({ theme }) => ({
  padding: theme.spacing(6, 0),
  background: "linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0a0a0a 100%)",
  position: "relative",
  overflow: "hidden",
  "&::before": {
    content: '""',
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background:
      "radial-gradient(circle at 50% 50%, rgba(255,215,0,0.05) 0%, transparent 70%)",
    pointerEvents: "none",
  },
}));

const SectionTitle = styled(Typography)(({ theme }) => ({
  color: "#fff",
  fontSize: "2.5rem",
  fontWeight: 900,
  textAlign: "center",
  marginBottom: theme.spacing(1),
  background: "linear-gradient(45deg, #FFD700, #FFA500, #FFD700)",
  backgroundClip: "text",
  WebkitBackgroundClip: "text",
  WebkitTextFillColor: "transparent",
  textShadow: "0 0 30px rgba(255, 215, 0, 0.5)",
}));

const SectionSubtitle = styled(Typography)(({ theme }) => ({
  color: "rgba(255,255,255,0.8)",
  fontSize: "1.1rem",
  textAlign: "center",
  marginBottom: theme.spacing(4),
  fontWeight: 300,
}));

const MoviesContainer = styled(Box)(() => ({
  position: "relative",
  width: "100%",
  overflow: "visible",
  paddingBottom: 20,
}));

const MoviesScroll = styled(Box)(({ theme }) => ({
  display: "flex",
  gap: theme.spacing(2),
  overflowX: "auto",
  scrollBehavior: "smooth",
  padding: theme.spacing(3, 1),
  // Hide scrollbar
  msOverflowStyle: "none",
  scrollbarWidth: "none",
  "&::-webkit-scrollbar": { display: "none" },
  // Allow overflow for hover effects
  overflowY: "visible",
}));

const NavigationButton = styled(IconButton)(({ theme, direction }) => ({
  position: "absolute",
  top: "50%",
  transform: "translateY(-50%)",
  [direction]: theme.spacing(2),
  zIndex: 10,
  background: "rgba(0,0,0,0.8)",
  color: "#FFD700",
  border: "2px solid rgba(255,215,0,0.3)",
  "&:hover": {
    background: "rgba(255,215,0,0.1)",
    border: "2px solid rgba(255,215,0,0.6)",
    animation: `${glowEffect} 1.5s ease-in-out infinite`,
  },
  "&:disabled": {
    opacity: 0.3,
  },
}));

const MovieCard = styled(Card)(() => ({
  minWidth: 280,
  maxWidth: 320,
  borderRadius: 20,
  overflow: "hidden",
  position: "relative",
  cursor: "pointer",
  background: "#0a0a0a",
  border: "1px solid rgba(255,215,0,0.1)",
  transition: "all 0.4s cubic-bezier(0.4, 0, 0.2, 1)",
  animation: `${slideIn} 0.6s ease-out`,
  "&:hover": {
    transform: "translateY(-8px) scale(1.02)",
    border: "1px solid rgba(255,215,0,0.4)",
    boxShadow: "0 20px 40px rgba(0,0,0,0.6), 0 0 20px rgba(255,215,0,0.2)",
    "& .hoverInfo": {
      opacity: 1,
      transform: "translateY(0)",
      pointerEvents: "auto",
    },
    "& .movieMedia": {
      transform: "scale(1.1)",
    },
  },
}));

const MovieMedia = styled(CardMedia)(() => ({
  height: 400,
  position: "relative",
  backgroundSize: "cover",
  backgroundPosition: "center",
  transition: "transform 0.4s ease",
}));

const HoverInfo = styled(Box)(({ theme }) => ({
  position: "absolute",
  bottom: 0,
  left: 0,
  right: 0,
  background: "linear-gradient(transparent, rgba(0,0,0,0.9))",
  color: "#fff",
  padding: theme.spacing(3, 2, 2),
  opacity: 0,
  transform: "translateY(40px)",
  transition: "all 0.4s cubic-bezier(0.4, 0, 0.2, 1)",
  pointerEvents: "none",
  zIndex: 2,
}));

const MovieTitle = styled(Typography)(() => ({
  fontSize: "1.2rem",
  fontWeight: 800,
  marginBottom: 8,
  background: "linear-gradient(45deg, #fff, #FFD700)",
  backgroundClip: "text",
  WebkitBackgroundClip: "text",
  WebkitTextFillColor: "transparent",
  lineHeight: 1.2,
}));

const MetaRow = styled(Box)(() => ({
  display: "flex",
  alignItems: "center",
  gap: 8,
  fontSize: "0.9rem",
  color: "rgba(255,255,255,0.9)",
  marginBottom: 12,
  fontWeight: 500,
}));

const GenreChip = styled(Chip)(() => ({
  background:
    "linear-gradient(45deg, rgba(255,215,0,0.2), rgba(255,165,0,0.2))",
  color: "#FFD700",
  fontSize: "0.75rem",
  height: 24,
  fontWeight: 600,
  border: "1px solid rgba(255,215,0,0.3)",
  "&:hover": {
    background:
      "linear-gradient(45deg, rgba(255,215,0,0.3), rgba(255,165,0,0.3))",
  },
}));

const Actions = styled(Box)(() => ({
  display: "flex",
  gap: 8,
  marginTop: 12,
}));

const ActionBtn = styled(IconButton)(() => ({
  color: "#fff",
  background: "rgba(255,255,255,0.15)",
  backdropFilter: "blur(10px)",
  border: "1px solid rgba(255,255,255,0.2)",
  transition: "all 0.3s ease",
  "&:hover": {
    color: "#FFD700",
    background: "rgba(255,215,0,0.2)",
    border: "1px solid rgba(255,215,0,0.5)",
    transform: "scale(1.1)",
    boxShadow: "0 4px 12px rgba(255,215,0,0.3)",
  },
}));

const FeaturedMovies = memo(function FeaturedMovies({
  movies,
  title = "Top 10 phim bộ hôm nay",
  subtitle,
}) {
  const navigate = useNavigate();
  const items = useMemo(() => (movies || []).slice(0, 10), [movies]);
  const scrollRef = useRef(null);

  const scrollLeft = () => {
    if (scrollRef.current) {
      // Lấy kích thước card đầu tiên + gap
      const firstCard = scrollRef.current.querySelector("[data-card]");
      if (firstCard) {
        const cardWidth = firstCard.offsetWidth;
        const gap = 16; // theme.spacing(2)
        scrollRef.current.scrollBy({
          left: -(cardWidth + gap),
          behavior: "smooth",
        });
      }
    }
  };

  const scrollRight = () => {
    if (scrollRef.current) {
      // Lấy kích thước card đầu tiên + gap
      const firstCard = scrollRef.current.querySelector("[data-card]");
      if (firstCard) {
        const cardWidth = firstCard.offsetWidth;
        const gap = 16; // theme.spacing(2)
        scrollRef.current.scrollBy({
          left: cardWidth + gap,
          behavior: "smooth",
        });
      }
    }
  };

  return (
    <SectionContainer>
      <Container maxWidth="xl">
        <SectionTitle
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 1.5,
          }}
        >
          <LocalMoviesIcon
            sx={{
              fontSize: 36,
              color: "#FFD700",
              filter: "drop-shadow(0 0 6px rgba(255,215,0,0.5))",
            }}
          />
          {title}
        </SectionTitle>
        {subtitle && <SectionSubtitle>{subtitle}</SectionSubtitle>}

        <MoviesContainer>
          <NavigationButton direction="left" onClick={scrollLeft} size="large">
            <ChevronLeftIcon fontSize="large" />
          </NavigationButton>

          <MoviesScroll ref={scrollRef}>
            {items.map((movie, index) => (
              <MovieCard
                key={movie.id}
                data-card
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <MovieMedia image={movie.thumb} className="movieMedia" />
                <HoverInfo className="hoverInfo">
                  <MovieTitle>{movie.title}</MovieTitle>
                  <MetaRow>
                    <StarIcon sx={{ fontSize: 18, color: "#FFD700" }} />
                    {movie.rating || "8.5"} • {movie.year || "2024"}
                  </MetaRow>

                  <Box
                    sx={{
                      display: "flex",
                      gap: 1,
                      flexWrap: "wrap",
                      marginBottom: 1,
                    }}
                  >
                    {movie.genres?.slice(0, 2).map((g, i) => (
                      <GenreChip
                        key={i}
                        icon={<LocalOfferIcon sx={{ fontSize: 16 }} />}
                        label={g}
                        size="small"
                      />
                    ))}
                  </Box>

                  <Actions>
                    <Tooltip title="Xem ngay" arrow>
                      <ActionBtn
                        size="small"
                        onClick={() =>
                          navigate(`/stream/${movie.id}`, {
                            state: { episode: 1, audioMode: "sub" },
                          })
                        }
                      >
                        <PlayArrowIcon />
                      </ActionBtn>
                    </Tooltip>
                    <Tooltip title="Yêu thích" arrow>
                      <ActionBtn
                        size="small"
                        onClick={async (e) => {
                          e.stopPropagation();
                          try {
                            await addFavoriteApi(movie.id);
                            window.dispatchEvent(
                              new CustomEvent("app:toast", {
                                detail: {
                                  message: "Đã thêm vào Yêu thích",
                                  severity: "success",
                                },
                              })
                            );
                          } catch (err) {
                            window.dispatchEvent(
                              new CustomEvent("app:toast", {
                                detail: {
                                  message: "Không thể thêm Yêu thích",
                                  severity: "error",
                                },
                              })
                            );
                          }
                        }}
                      >
                        <FavoriteBorderIcon />
                      </ActionBtn>
                    </Tooltip>
                    <Tooltip title="Chi tiết" arrow>
                      <ActionBtn
                        size="small"
                        onClick={() => navigate(`/movie/${movie.id}`)}
                      >
                        <InfoOutlinedIcon />
                      </ActionBtn>
                    </Tooltip>
                  </Actions>
                </HoverInfo>
              </MovieCard>
            ))}
          </MoviesScroll>

          <NavigationButton
            direction="right"
            onClick={scrollRight}
            size="large"
          >
            <ChevronRightIcon fontSize="large" />
          </NavigationButton>
        </MoviesContainer>
      </Container>
    </SectionContainer>
  );
});

FeaturedMovies.propTypes = {
  movies: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
      title: PropTypes.string.isRequired,
      thumb: PropTypes.string.isRequired,
      rating: PropTypes.string,
      year: PropTypes.string,
      genres: PropTypes.arrayOf(PropTypes.string),
    })
  ).isRequired,
  title: PropTypes.string,
  subtitle: PropTypes.string,
};

export default FeaturedMovies;
