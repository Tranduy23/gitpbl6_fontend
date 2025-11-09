/* eslint-disable react/display-name */
import { Suspense, lazy } from "react";
import { Navigate, useRoutes, Outlet } from "react-router-dom";
// components
import Loading from "../components/Loading";

// ----------------------------------------------------------------------

const Loadable = (Component) => (props) => {
  return (
    <Suspense fallback={<Loading message="Đang tải trang..." />}>
      <Component {...props} />
    </Suspense>
  );
};

export default function Router() {
  return useRoutes([
    {
      path: "/",
      element: (
        <div>
          <Outlet />
        </div>
      ),
      children: [
        { element: <HomePage />, index: true },
        { path: "movies", element: <MoviesPage /> },
        { path: "movie/:id", element: <MovieDetailPage /> },
        { path: "actors", element: <ActorsPage /> },
        { path: "actor/:id", element: <ActorDetailPage /> },
        { path: "stream/:id", element: <StreamPage /> },
        { path: "showtime", element: <ShowtimePage /> },
        { path: "login", element: <LoginPage /> },
        { path: "register", element: <RegisterPage /> },
        { path: "favorites", element: <FavoritesPage /> },
        { path: "watchlist", element: <WatchlistPage /> },
        { path: "watch-history", element: <WatchHistoryPage /> },
        { path: "account", element: <AccountPage /> },
        { path: "admin", element: <AdminManagementPage /> },
      ],
    },
    { path: "*", element: <Navigate to="/404" replace /> },
  ]);
}

// IMPORT COMPONENTS

const HomePage = Loadable(lazy(() => import("../pages/Home")));
const MoviesPage = Loadable(lazy(() => import("../pages/Movies")));
const MovieDetailPage = Loadable(lazy(() => import("../pages/MovieDetail")));
const ActorsPage = Loadable(lazy(() => import("../pages/Actors")));
const ActorDetailPage = Loadable(lazy(() => import("../pages/ActorDetail")));
const StreamPage = Loadable(lazy(() => import("../pages/Stream")));
const ShowtimePage = Loadable(lazy(() => import("../pages/Showtime")));
const LoginPage = Loadable(lazy(() => import("../pages/Login")));
const RegisterPage = Loadable(lazy(() => import("../pages/Register")));
const FavoritesPage = Loadable(lazy(() => import("../pages/Favorites")));
const WatchlistPage = Loadable(lazy(() => import("../pages/Watchlist")));
const WatchHistoryPage = Loadable(lazy(() => import("../pages/WatchHistory")));
const AccountPage = Loadable(lazy(() => import("../pages/Account")));
const AdminManagementPage = Loadable(
  lazy(() => import("../pages/AdminManagement"))
);
