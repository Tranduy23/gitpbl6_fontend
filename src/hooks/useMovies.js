import { useState, useEffect } from "react";

const useMovies = () => {
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchMovies = async () => {
      try {
        setLoading(true);
        // Use Vite proxy to avoid CORS issues
        const response = await fetch(
          `/api/movies?page=0&size=50&sort=title,asc`,
          {
            method: "GET",
            headers: {
              Accept: "*/*",
              "Content-Type": "application/json",
            },
          }
        );

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Transform API data to match the expected format
        const transformedMovies = data.map((movie) => ({
          id: movie.id,
          title: movie.title,
          englishTitle: movie.title,
          ageRating: movie.ageRating || null,
          imdbRating: movie.imdbRating ?? null,
          averageRating: movie.averageRating ?? null,
          totalRatings: movie.totalRatings ?? 0,
          year: movie.year ? String(movie.year) : undefined,
          duration: movie.videoDuration || undefined,
          genres: movie.categories || [],
          synopsis: movie.synopsis || "",
          thumb: movie.thumbnailUrl || movie.posterUrl || undefined,
          thumbnailUrl: movie.thumbnailUrl || undefined,
          posterUrl: movie.posterUrl || undefined,
          videoUrl: movie.videoUrl || undefined,
          trailerUrl: movie.trailerUrl || undefined,
          streamingUrl: movie.streamingUrl || undefined,
          isAvailable: Boolean(movie.isAvailable),
          actors: movie.actors || [],
          directors: movie.directors || [],
          directorName: movie.directorName || undefined,
          country: movie.country || undefined,
          language: movie.language || undefined,
          viewCount: movie.viewCount ?? 0,
          likeCount: movie.likeCount ?? 0,
          dislikeCount: movie.dislikeCount ?? 0,
          isFeatured: Boolean(movie.isFeatured),
          isTrending: Boolean(movie.isTrending),
          releaseDate: movie.releaseDate || undefined,
          downloadEnabled: Boolean(movie.downloadEnabled),
          availableQualities: movie.availableQualities || [],
        }));

        setMovies(transformedMovies);
      } catch (err) {
        console.error("Error fetching movies:", err);
        setError(err.message);

        // Fallback to mock data if API fails
        setMovies([
          {
            id: 1,
            title: "Chuyện tình tóc rối",
            englishTitle: "Love Untangled",
            rating: "T13",
            year: "2025",
            duration: "1h 58m",
            genres: ["Chính Kịch", "Hài", "Tuổi Trẻ", "Học Đường", "Lãng Mạn"],
            synopsis:
              "Một thiếu nữ si tình lên kế hoạch chinh phục nam thần của trường bằng cách chuyển từ tóc xoăn sang tóc thẳng – cho đến khi một học sinh mới chuyển đến thay đổi mọi chuyện.",
            thumb:
              "https://occ-0-8407-92.1.nflxso.net/dnm/api/v6/E8vDc_W8CLv7-yMQu8KMEC7Rrr8/AAAABVA4RTUHTxnwFesffshDCUKcSlGkUGAJclgeOkoj_WwrzBvLOjVztCJlElXplZkRQVyAi3cjePjYv5xSO12xxsztsGGpjwBKeSeV.jpg?r=d74",
          },
          {
            id: 2,
            title: "Mùa Hè Kinh Hãi",
            englishTitle: "I Know What You Did Last Summer",
            rating: "T18",
            year: "2025",
            duration: "1h 51m",
            genres: ["Chiếu Rạp", "Gay Cấn", "Kinh Dị", "Bí Ẩn", "Tâm Lý"],
            synopsis:
              "Khi năm người bạn vô tình gây ra một vụ tai nạn xe hơi chết người, họ quyết định che giấu…",
            thumb: "https://picsum.photos/400/600?random=2",
          },
          {
            id: 3,
            title: "Romantic Drama",
            englishTitle: "A Quiet Glance",
            rating: "T13",
            year: "2023",
            duration: "1h 45m",
            genres: ["Tâm lý", "Lãng mạn"],
            synopsis: "Hai con người cô độc tìm thấy nhau…",
            thumb: "https://picsum.photos/400/600?random=3",
          },
          {
            id: 4,
            title: "Cartoon Adventure",
            englishTitle: "Splash Planet",
            rating: "P",
            year: "2022",
            duration: "1h 32m",
            genres: ["Hoạt hình", "Phiêu lưu"],
            synopsis: "Những người bạn nhỏ vượt đại dương…",
            thumb: "https://picsum.photos/400/600?random=4",
          },
          {
            id: 5,
            title: "Dramatic Landscape",
            englishTitle: "Sunset Letters",
            rating: "T13",
            year: "2021",
            duration: "1h 57m",
            genres: ["Chính kịch"],
            synopsis: "Những bức thư cũ mở ra câu chuyện dang dở…",
            thumb: "https://picsum.photos/400/600?random=5",
          },
          {
            id: 6,
            title: "Horror Thriller",
            englishTitle: "The Sixth Door",
            rating: "T18",
            year: "2024",
            duration: "1h 49m",
            genres: ["Kinh dị", "Giật gân"],
            synopsis: "Cánh cửa thứ sáu chỉ mở khi ai đó dám nhìn vào nỗi sợ…",
            thumb: "https://picsum.photos/400/600?random=6",
          },
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchMovies();
  }, []);

  return { movies, loading, error };
};

export default useMovies;
