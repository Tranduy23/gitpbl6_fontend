import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Box, Container, Grid, Typography, Chip, Stack } from "@mui/material";
import { styled } from "@mui/material/styles";
import { getActorById, listMoviesOfActor } from "../api/actors";
import Header from "../components/Header";

const PageContainer = styled(Box)(() => ({
  minHeight: "100vh",
  background: "linear-gradient(180deg, #0b1220 0%, #0a0f1a 50%, #0b0d13 100%)",
  color: "#fff",
  paddingTop: 96,
}));

const Card = styled(Box)(() => ({
  background: "rgba(255,255,255,0.05)",
  border: "1px solid rgba(255,255,255,0.12)",
  borderRadius: 12,
  overflow: "hidden",
}));

export default function ActorDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [actor, setActor] = useState(null);
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        setError("");
        const [a, m] = await Promise.all([
          getActorById(id),
          listMoviesOfActor(id, { page: 0, size: 24 }),
        ]);
        setActor(a);
        const list = Array.isArray(m?.content)
          ? m.content
          : Array.isArray(m)
          ? m
          : [];
        setMovies(list);
      } catch (e) {
        setError(e?.message || "Không thể tải thông tin diễn viên");
      } finally {
        setLoading(false);
      }
    })();
  }, [id]);

  const openMovie = (movieId) => navigate(`/movie/${movieId}`);

  return (
    <PageContainer>
      <Header />
      <Container maxWidth="xl" sx={{ pt: 2, pb: 6 }}>
        {loading && !actor && (
          <Typography sx={{ color: "#fff", mb: 2 }}>Đang tải...</Typography>
        )}
        {error && (
          <Typography sx={{ color: "#ff6b6b", mb: 2 }}>{error}</Typography>
        )}

        <Grid container spacing={5}>
          {/* Left column: actor info */}
          <Grid item xs={12} md={3} lg={3} sx={{ pr: { md: 4 } }}>
            {actor && (
              <Box>
                <Card>
                  <Box
                    component="img"
                    src={actor.imageUrl || actor.image || actor.avatarUrl}
                    alt={actor.name}
                    onError={(e) => {
                      e.currentTarget.src = `https://via.placeholder.com/400x500/333/fff?text=${encodeURIComponent(
                        actor.name || "Actor"
                      )}`;
                    }}
                    sx={{ width: "100%", height: 440, objectFit: "cover" }}
                  />
                </Card>

                <Typography variant="h4" sx={{ fontWeight: 800, mt: 2, mb: 1 }}>
                  {actor?.name}
                </Typography>
                {actor?.dob && (
                  <Typography sx={{ opacity: 0.8, mb: 1 }}>
                    Sinh: {actor.dob}
                  </Typography>
                )}
                {actor?.description && (
                  <Typography sx={{ opacity: 0.9 }}>
                    {actor.description}
                  </Typography>
                )}

                {Array.isArray(actor?.alsoKnownAs) &&
                  actor.alsoKnownAs.length > 0 && (
                    <Box sx={{ mt: 2 }}>
                      <Typography sx={{ opacity: 0.8, mb: 1 }}>
                        Tên gọi khác:
                      </Typography>
                      <Stack
                        direction="row"
                        spacing={1}
                        flexWrap="wrap"
                        useFlexGap
                      >
                        {actor.alsoKnownAs.map((n) => (
                          <Chip key={n} label={n} size="small" sx={{ mb: 1 }} />
                        ))}
                      </Stack>
                    </Box>
                  )}
              </Box>
            )}
          </Grid>

          {/* Right column: movies grid */}
          <Grid
            item
            xs={12}
            md={9}
            lg={9}
            sx={{
              pl: { md: 4 },
              borderLeft: { md: "1px solid rgba(255,255,255,0.1)" },
            }}
          >
            <Typography variant="h5" sx={{ fontWeight: 800, mb: 2 }}>
              Phim đã tham gia
            </Typography>
            <Grid container spacing={3}>
              {movies.map((mv) => (
                <Grid item xs={6} sm={4} md={3} key={mv.id}>
                  <Card
                    onClick={() => openMovie(mv.id)}
                    sx={{ cursor: "pointer" }}
                  >
                    <Box
                      component="img"
                      src={mv.posterUrl || mv.poster}
                      alt={mv.title}
                      onError={(e) => {
                        e.currentTarget.src = `https://via.placeholder.com/400x500/333/fff?text=${encodeURIComponent(
                          mv.title || "Movie"
                        )}`;
                      }}
                      sx={{
                        width: "100%",
                        height: 300,
                        objectFit: "cover",
                        display: "block",
                      }}
                    />
                    <Box sx={{ p: 1.5 }}>
                      <Typography sx={{ fontWeight: 700, mb: 0.5 }} noWrap>
                        {mv.title}
                      </Typography>
                      <Stack direction="row" spacing={1} alignItems="center">
                        {mv.imdbRating && (
                          <Typography sx={{ fontSize: 14, opacity: 0.9 }}>
                            ⭐ {mv.imdbRating}
                          </Typography>
                        )}
                        {mv.year && (
                          <Typography sx={{ fontSize: 14, opacity: 0.7 }}>
                            • {mv.year}
                          </Typography>
                        )}
                      </Stack>
                    </Box>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Grid>
        </Grid>
      </Container>
    </PageContainer>
  );
}
