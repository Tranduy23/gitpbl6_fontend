import { useState, useEffect, useRef } from "react";
import {
  Box,
  Container,
  Typography,
  Tabs,
  Tab,
  Card,
  CardContent,
  Grid,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Alert,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Avatar,
  CardMedia,
  CardActions,
  CardHeader,
  Stack,
  Tooltip,
  Autocomplete,
  Pagination,
} from "@mui/material";
import TmdbMovieSearch from "../components/TmdbMovieSearch";
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Visibility as ViewIcon,
  Dashboard as DashboardIcon,
  TrendingUp as TrendingIcon,
  Person as PersonIcon,
  Movie as MovieIcon,
  ArrowBack as ArrowBackIcon,
  LocalMovies as MoviesIcon,
  People as UsersIcon,
  Assessment as StatisticsIcon,
  Report as ReportsIcon,
  Public as PublicIcon,
  CloudUpload as CloudUploadIcon,
  ViewList as ViewListIcon,
  ViewModule as ViewModuleIcon,
  TheaterComedy as TheaterComedyIcon,
  Category as CategoryIcon,
  ClosedCaption as ClosedCaptionIcon,
} from "@mui/icons-material";
import * as adminAPI from "../api/admin";
import AdminRoute from "../components/AdminRoute";
import { useNavigate } from "react-router-dom";

// Tab panel component
function TabPanel({ children, value, index, ...other }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`admin-tabpanel-${index}`}
      aria-labelledby={`admin-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

// Directors management component (list + create/edit)
function DirectorsManagement() {
  const [directors, setDirectors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [editing, setEditing] = useState(null);
  const [viewing, setViewing] = useState(null);
  const [viewMode, setViewMode] = useState("grid");
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    loadDirectors();
  }, []);

  const loadDirectors = async () => {
    try {
      setLoading(true);
      let res = null;
      try {
        res = await adminAPI.getAllDirectors();
      } catch (_) {
        res = await adminAPI.getDirectors();
      }
      const listCandidate = Array.isArray(res)
        ? res
        : res?.data ?? res?.content ?? res?.data?.content ?? [];
      setDirectors(Array.isArray(listCandidate) ? listCandidate : []);
    } catch (err) {
      setError(err.message || "Failed to load directors");
    } finally {
      setLoading(false);
    }
  };

  const handleSaved = () => {
    setShowCreate(false);
    setEditing(null);
    setSuccess("Director saved successfully");
    loadDirectors();
  };

  if (viewing) {
    return (
      <DirectorViewForm director={viewing} onClose={() => setViewing(null)} />
    );
  }

  return (
    <Box>
      {success && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess("")}>
          {success}
        </Alert>
      )}
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          mb: 3,
        }}
      >
        <Typography variant="h5">Directors</Typography>
        <Box sx={{ display: "flex", gap: 2 }}>
          <Button
            variant="outlined"
            onClick={() => setViewMode(viewMode === "grid" ? "table" : "grid")}
            startIcon={
              viewMode === "grid" ? <ViewListIcon /> : <ViewModuleIcon />
            }
          >
            {viewMode === "grid" ? "Table View" : "Grid View"}
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setShowCreate(true)}
          >
            Add Director
          </Button>
        </Box>
      </Box>

      {/* Search */}
      <Box sx={{ mb: 3 }}>
        <TextField
          placeholder="Search directors by name..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          size="small"
          sx={{ minWidth: 300 }}
          fullWidth
        />
      </Box>

      {showCreate || editing ? (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              {editing ? "Edit Director" : "Create Director"}
            </Typography>
            <DirectorForm
              initialData={editing}
              onCancel={() => {
                setShowCreate(false);
                setEditing(null);
              }}
              onSuccess={handleSaved}
            />
          </CardContent>
        </Card>
      ) : loading ? (
        <Box sx={{ display: "flex", justifyContent: "center", p: 3 }}>
          <CircularProgress />
        </Box>
      ) : error ? (
        <Alert severity="error">{error}</Alert>
      ) : (
        (() => {
          const items = (Array.isArray(directors) ? directors : []).filter(
            (d) =>
              d &&
              d.name &&
              String(d.name).trim().length > 0 &&
              String(d.name).toLowerCase() !== "unknown director" &&
              (searchTerm === "" ||
                String(d.name).toLowerCase().includes(searchTerm.toLowerCase()))
          );
          if (items.length === 0) return null; // don't render anything when empty
          return viewMode === "grid" ? (
            <Grid container spacing={3}>
              {items.map((d) => (
                <Grid item xs={12} sm={6} md={4} lg={3} key={d.id}>
                  <Card>
                    <CardContent>
                      <Box sx={{ textAlign: "center", mb: 1 }}>
                        <Avatar
                          src={d.photoUrl || d.photourl}
                          sx={{ width: 80, height: 80, mx: "auto" }}
                        />
                      </Box>
                      <Typography variant="h6" gutterBottom>
                        {d.name}
                      </Typography>
                      {d.birthDate && (
                        <Typography variant="caption" color="text.secondary">
                          DOB: {d.birthDate}
                        </Typography>
                      )}
                    </CardContent>
                    <CardActions
                      sx={{ justifyContent: "space-between", px: 2, pb: 2 }}
                    >
                      <Tooltip title="View">
                        <IconButton
                          size="small"
                          color="primary"
                          onClick={() => setViewing(d)}
                        >
                          <ViewIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Edit">
                        <IconButton
                          size="small"
                          color="secondary"
                          onClick={() => setEditing(d)}
                        >
                          <EditIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Delete">
                        <IconButton
                          size="small"
                          color="error"
                          onClick={async () => {
                            try {
                              await adminAPI.deleteDirector(d.id);
                              setSuccess("Director deleted");
                              loadDirectors();
                            } catch (err) {
                              setError(
                                err.message || "Failed to delete director"
                              );
                            }
                          }}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </Tooltip>
                    </CardActions>
                  </Card>
                </Grid>
              ))}
            </Grid>
          ) : (
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Photo</TableCell>
                    <TableCell>Name</TableCell>
                    <TableCell>DOB</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {items.map((d) => (
                    <TableRow key={d.id} hover>
                      <TableCell>
                        <Avatar
                          src={d.photoUrl || d.photourl}
                          sx={{ width: 50, height: 50 }}
                        />
                      </TableCell>
                      <TableCell>{d.name}</TableCell>
                      <TableCell>{d.birthDate || ""}</TableCell>
                      <TableCell>
                        <Box sx={{ display: "flex", gap: 0.5 }}>
                          <Tooltip title="View">
                            <IconButton
                              size="small"
                              color="primary"
                              onClick={() => setViewing(d)}
                            >
                              <ViewIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Edit">
                            <IconButton
                              size="small"
                              color="secondary"
                              onClick={() => setEditing(d)}
                            >
                              <EditIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete">
                            <IconButton
                              size="small"
                              color="error"
                              onClick={async () => {
                                try {
                                  await adminAPI.deleteDirector(d.id);
                                  setSuccess("Director deleted");
                                  loadDirectors();
                                } catch (err) {
                                  setError(
                                    err.message || "Failed to delete director"
                                  );
                                }
                              }}
                            >
                              <DeleteIcon />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          );
        })()
      )}
    </Box>
  );
}

function DirectorViewForm({ director, onClose }) {
  const [details, setDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    load();
  }, [director.id]);

  const load = async () => {
    try {
      setLoading(true);
      const res = await adminAPI.getDirectorById(director.id);
      setDetails(res?.data || res);
    } catch (err) {
      setError(err.message || "Failed to load director");
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ display: "flex", justifyContent: "center", p: 3 }}>
            <CircularProgress />
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Alert severity="error">{error}</Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            mb: 3,
          }}
        >
          <Typography variant="h6">
            Director Details: {details?.name}
          </Typography>
          <Button onClick={onClose} variant="outlined">
            Close
          </Button>
        </Box>

        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Box sx={{ textAlign: "center" }}>
              <Avatar
                src={details?.photoUrl || details?.photourl}
                sx={{ width: 120, height: 120, mx: "auto" }}
              />
            </Box>
          </Grid>
          <Grid item xs={12} md={8}>
            <Typography variant="h5" gutterBottom>
              {details?.name}
            </Typography>
            {details?.birthDate && (
              <Typography variant="body2" color="text.secondary" gutterBottom>
                DOB: {details.birthDate}
              </Typography>
            )}
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Biography
              </Typography>
              <Typography variant="body1">
                {details?.biography || details?.bio || details?.description}
              </Typography>
            </Box>
            {typeof details?.movieCount === "number" && (
              <Chip label={`Movies: ${details.movieCount}`} color="primary" />
            )}
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
}

function DirectorForm({ initialData, onSuccess, onCancel }) {
  const [formData, setFormData] = useState({
    name: initialData?.name || "",
    biography:
      initialData?.biography ||
      initialData?.bio ||
      initialData?.description ||
      "",
    birthDate: initialData?.birthDate || "",
    nationality: initialData?.nationality || "",
  });
  const [photo, setPhoto] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (field) => (e) =>
    setFormData((p) => ({ ...p, [field]: e.target.value }));
  const handlePhoto = (e) => setPhoto(e.target.files?.[0] || null);

  const onSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setError("");
    try {
      if (initialData?.id) {
        // If a new photo is selected, use multipart update
        if (photo) {
          await adminAPI.updateDirectorWithFormData(initialData.id, {
            ...formData,
            photo,
          });
        } else {
          await adminAPI.updateDirector(initialData.id, formData);
        }
      } else {
        await adminAPI.createDirector({ ...formData, photo });
      }
      onSuccess?.();
    } catch (err) {
      setError(err.message || "Failed to save director");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Box component="form" onSubmit={onSubmit}>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            label="Name"
            value={formData.name}
            onChange={handleChange("name")}
            required
          />
        </Grid>
        <Grid item xs={12}>
          <TextField
            fullWidth
            multiline
            rows={3}
            label="Biography"
            value={formData.biography}
            onChange={handleChange("biography")}
          />
        </Grid>
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            type="date"
            InputLabelProps={{ shrink: true }}
            label="Birth Date"
            value={formData.birthDate}
            onChange={handleChange("birthDate")}
          />
        </Grid>
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            label="Nationality"
            value={formData.nationality}
            onChange={handleChange("nationality")}
          />
        </Grid>

        <Grid item xs={12}>
          <input
            id="director-photo"
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={handlePhoto}
          />
          <label htmlFor="director-photo">
            <Button
              variant="outlined"
              component="span"
              startIcon={<CloudUploadIcon />}
            >
              {photo
                ? photo.name
                : initialData?.id
                ? "Change Photo"
                : "Upload Photo"}
            </Button>
          </label>
        </Grid>

        <Grid item xs={12}>
          <Box sx={{ display: "flex", gap: 2, justifyContent: "flex-end" }}>
            <Button onClick={onCancel} disabled={submitting}>
              Cancel
            </Button>
            <Button
              type="submit"
              variant="contained"
              disabled={submitting}
              startIcon={
                submitting ? <CircularProgress size={18} /> : <AddIcon />
              }
            >
              {initialData?.id ? "Update" : "Create"}
            </Button>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
}

// Actors management component (list + create/edit/view)
function CategoriesManagement() {
  const [categories, setCategories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [editing, setEditing] = useState(null);
  const [viewMode, setViewMode] = useState("grid");
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    loadCategories();
  }, []);

  const loadCategories = async () => {
    try {
      setLoading(true);
      const res = await adminAPI.getCategories(0, 20);
      const list = Array.isArray(res) ? res : res?.content || res?.data || [];
      setCategories(list);
    } catch (err) {
      setError(err.message || "Failed to load categories");
    } finally {
      setLoading(false);
    }
  };

  const handleSaved = () => {
    setShowCreate(false);
    setEditing(null);
    setSuccess("Category saved successfully");
    loadCategories();
  };

  const handleDelete = async (name) => {
    try {
      await adminAPI.deleteCategory(name);
      setSuccess("Category deleted");
      loadCategories();
    } catch (err) {
      setError(err.message || "Failed to delete category");
    }
  };

  return (
    <Box>
      {success && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess("")}>
          {success}
        </Alert>
      )}
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          mb: 3,
        }}
      >
        <Typography variant="h5">Categories</Typography>
        <Box sx={{ display: "flex", gap: 2 }}>
          <Button
            variant="outlined"
            onClick={() => setViewMode(viewMode === "grid" ? "table" : "grid")}
            startIcon={
              viewMode === "grid" ? <ViewListIcon /> : <ViewModuleIcon />
            }
          >
            {viewMode === "grid" ? "Table View" : "Grid View"}
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setShowCreate(true)}
          >
            Add Category
          </Button>
        </Box>
      </Box>

      {/* Search */}
      <Box sx={{ mb: 3 }}>
        <TextField
          placeholder="Search categories by name..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          size="small"
          sx={{ minWidth: 300 }}
          fullWidth
        />
      </Box>

      {showCreate || editing ? (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              {editing ? "Edit Category" : "Create Category"}
            </Typography>
            <CategoryForm
              initialData={editing}
              onCancel={() => {
                setShowCreate(false);
                setEditing(null);
              }}
              onSuccess={handleSaved}
            />
          </CardContent>
        </Card>
      ) : loading ? (
        <Box sx={{ display: "flex", justifyContent: "center", p: 3 }}>
          <CircularProgress />
        </Box>
      ) : error ? (
        <Alert severity="error">{error}</Alert>
      ) : viewMode === "grid" ? (
        <Grid container spacing={3}>
          {(Array.isArray(categories) ? categories : [])
            .filter(
              (c) =>
                searchTerm === "" ||
                String(c.name || "")
                  .toLowerCase()
                  .includes(searchTerm.toLowerCase()) ||
                String(c.displayName || "")
                  .toLowerCase()
                  .includes(searchTerm.toLowerCase())
            )
            .map((c) => (
              <Grid item xs={12} sm={6} md={4} lg={3} key={c.id || c.name}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      {c.displayName || c.name}
                    </Typography>
                    {c.description && (
                      <Typography variant="body2" color="text.secondary">
                        {c.description}
                      </Typography>
                    )}
                    {typeof c.movieCount === "number" && (
                      <Chip
                        label={`Movies: ${c.movieCount}`}
                        size="small"
                        sx={{ mt: 1 }}
                      />
                    )}
                  </CardContent>
                  <CardActions
                    sx={{ justifyContent: "space-between", px: 2, pb: 2 }}
                  >
                    <Tooltip title="Assign Movie (demo: movie id 1)">
                      <IconButton
                        size="small"
                        color="primary"
                        onClick={async () => {
                          try {
                            await adminAPI.assignMovieToCategory(
                              c.name || c.displayName,
                              1
                            );
                            setSuccess("Assigned movie to category");
                            loadCategories();
                          } catch (err) {
                            setError(err.message || "Failed to assign movie");
                          }
                        }}
                      >
                        <AddIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Edit">
                      <IconButton
                        size="small"
                        color="secondary"
                        onClick={() => setEditing(c)}
                      >
                        <EditIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete">
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => handleDelete(c.name || c.displayName)}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Tooltip>
                  </CardActions>
                </Card>
              </Grid>
            ))}
        </Grid>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Display Name</TableCell>
                <TableCell>Description</TableCell>
                <TableCell>Movies</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {(Array.isArray(categories) ? categories : [])
                .filter(
                  (c) =>
                    searchTerm === "" ||
                    String(c.name || "")
                      .toLowerCase()
                      .includes(searchTerm.toLowerCase()) ||
                    String(c.displayName || "")
                      .toLowerCase()
                      .includes(searchTerm.toLowerCase())
                )
                .map((c) => (
                  <TableRow key={c.id || c.name} hover>
                    <TableCell>{c.name}</TableCell>
                    <TableCell>{c.displayName}</TableCell>
                    <TableCell>
                      <Typography variant="body2" noWrap sx={{ maxWidth: 240 }}>
                        {c.description}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      {typeof c.movieCount === "number" && (
                        <Chip label={c.movieCount} size="small" />
                      )}
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: "flex", gap: 0.5 }}>
                        <Tooltip title="Assign Movie (demo: movie id 1)">
                          <IconButton
                            size="small"
                            color="primary"
                            onClick={async () => {
                              try {
                                await adminAPI.assignMovieToCategory(
                                  c.name || c.displayName,
                                  1
                                );
                                setSuccess("Assigned movie to category");
                                loadCategories();
                              } catch (err) {
                                setError(
                                  err.message || "Failed to assign movie"
                                );
                              }
                            }}
                          >
                            <AddIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Edit">
                          <IconButton
                            size="small"
                            color="secondary"
                            onClick={() => setEditing(c)}
                          >
                            <EditIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Delete">
                          <IconButton
                            size="small"
                            color="error"
                            onClick={() =>
                              handleDelete(c.name || c.displayName)
                            }
                          >
                            <DeleteIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
}

function CategoryForm({ initialData, onSuccess, onCancel }) {
  const [formData, setFormData] = useState({
    name: initialData?.name || "",
    displayName: initialData?.displayName || "",
    description: initialData?.description || "",
    icon: initialData?.icon || "",
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (field) => (e) =>
    setFormData((p) => ({ ...p, [field]: e.target.value }));

  const onSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setError("");
    try {
      if (initialData?.name) {
        await adminAPI.updateCategory(initialData.name, formData);
      } else {
        await adminAPI.createCategory(formData);
      }
      onSuccess?.();
    } catch (err) {
      setError(err.message || "Failed to save category");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Box component="form" onSubmit={onSubmit}>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            label="Name"
            value={formData.name}
            onChange={handleChange("name")}
            required
            disabled={!!initialData?.name}
          />
        </Grid>
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            label="Display Name"
            value={formData.displayName}
            onChange={handleChange("displayName")}
          />
        </Grid>
        <Grid item xs={12}>
          <TextField
            fullWidth
            multiline
            rows={3}
            label="Description"
            value={formData.description}
            onChange={handleChange("description")}
          />
        </Grid>
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            label="Icon"
            value={formData.icon}
            onChange={handleChange("icon")}
          />
        </Grid>
        <Grid item xs={12}>
          <Box sx={{ display: "flex", gap: 2, justifyContent: "flex-end" }}>
            <Button onClick={onCancel} disabled={submitting}>
              Cancel
            </Button>
            <Button
              type="submit"
              variant="contained"
              disabled={submitting}
              startIcon={
                submitting ? <CircularProgress size={18} /> : <AddIcon />
              }
            >
              {initialData?.name ? "Update" : "Create"}
            </Button>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
}
function ActorsManagement() {
  const [actors, setActors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [editing, setEditing] = useState(null);
  const [viewing, setViewing] = useState(null);
  const [page, setPage] = useState(0);
  const [size] = useState(20);
  const [viewMode, setViewMode] = useState("grid");
  const [totalPages, setTotalPages] = useState(0);
  const [totalElements, setTotalElements] = useState(0);
  const [usePagination, setUsePagination] = useState(true); // Toggle pagination
  const [allActors, setAllActors] = useState([]); // Store all actors for client-side pagination
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    loadActors();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, usePagination]);

  const loadActors = async () => {
    try {
      setLoading(true);
      setError("");
      let res;

      if (usePagination) {
        try {
          // Try with pagination first
          res = await adminAPI.getActors(page, size);
        } catch (paginationError) {
          // If pagination fails, disable it and load all
          console.warn(
            "Pagination failed, loading all actors:",
            paginationError
          );
          setUsePagination(false);
          res = await adminAPI.getActors(0, 1000);
        }
      } else {
        // Load all without pagination (only once)
        if (allActors.length === 0) {
          res = await adminAPI.getActors(0, 1000);
        } else {
          // Use cached data for client-side pagination
          const startIndex = page * size;
          const endIndex = startIndex + size;
          setActors(allActors.slice(startIndex, endIndex));
          setTotalPages(Math.ceil(allActors.length / size));
          setTotalElements(allActors.length);
          setLoading(false);
          return;
        }
      }

      // Handle paginated response
      if (res?.content) {
        const allActors = res.content;
        if (usePagination) {
          setActors(allActors);
          setTotalPages(res.totalPages || 0);
          setTotalElements(res.totalElements || 0);
        } else {
          // Client-side pagination
          setAllActors(allActors);
          const startIndex = page * size;
          const endIndex = startIndex + size;
          setActors(allActors.slice(startIndex, endIndex));
          setTotalPages(Math.ceil(allActors.length / size));
          setTotalElements(allActors.length);
        }
      } else if (res?.data?.content) {
        const allActorsList = res.data.content;
        if (usePagination) {
          setActors(allActorsList);
          setTotalPages(res.data.totalPages || 0);
          setTotalElements(res.data.totalElements || 0);
        } else {
          setAllActors(allActorsList);
          const startIndex = page * size;
          const endIndex = startIndex + size;
          setActors(allActorsList.slice(startIndex, endIndex));
          setTotalPages(Math.ceil(allActorsList.length / size));
          setTotalElements(allActorsList.length);
        }
      } else {
        const list = Array.isArray(res) ? res : res?.data || [];
        if (usePagination) {
          setActors(list);
          setTotalPages(1);
          setTotalElements(list.length);
        } else {
          setAllActors(list);
          const startIndex = page * size;
          const endIndex = startIndex + size;
          setActors(list.slice(startIndex, endIndex));
          setTotalPages(Math.ceil(list.length / size));
          setTotalElements(list.length);
        }
      }
    } catch (err) {
      console.error("Error loading actors:", err);
      setError(err.message || "Failed to load actors");
      setActors([]);
      setTotalPages(0);
      setTotalElements(0);
    } finally {
      setLoading(false);
    }
  };

  const handlePageChange = (event, newPage) => {
    setPage(newPage - 1); // MUI Pagination uses 1-based indexing
  };

  const handleCreated = () => {
    setShowCreate(false);
    setEditing(null);
    setSuccess("Actor saved successfully");
    setAllActors([]); // Clear cache to reload
    setPage(0); // Reset to first page
    loadActors();
  };

  const handleDeleted = async (actor) => {
    try {
      await adminAPI.deleteActor(actor.id);
      setSuccess("Actor deleted");
      setAllActors([]); // Clear cache to reload
      loadActors();
    } catch (err) {
      setError(err.message || "Failed to delete actor");
    }
  };

  if (viewing) {
    return <ActorViewForm actor={viewing} onClose={() => setViewing(null)} />;
  }

  return (
    <Box>
      {success && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess("")}>
          {success}
        </Alert>
      )}
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          mb: 3,
        }}
      >
        <Typography variant="h5">Actors</Typography>
        <Box sx={{ display: "flex", gap: 2 }}>
          <Button
            variant="outlined"
            onClick={() => setViewMode(viewMode === "grid" ? "table" : "grid")}
            startIcon={
              viewMode === "grid" ? <ViewListIcon /> : <ViewModuleIcon />
            }
          >
            {viewMode === "grid" ? "Table View" : "Grid View"}
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setShowCreate(true)}
          >
            Add Actor
          </Button>
        </Box>
      </Box>

      {/* Search */}
      <Box sx={{ mb: 3 }}>
        <TextField
          placeholder="Search actors by name..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          size="small"
          sx={{ minWidth: 300 }}
          fullWidth
        />
      </Box>

      {showCreate || editing ? (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              {editing ? "Edit Actor" : "Create Actor"}
            </Typography>
            <ActorForm
              initialData={editing}
              onCancel={() => {
                setShowCreate(false);
                setEditing(null);
              }}
              onSuccess={handleCreated}
            />
          </CardContent>
        </Card>
      ) : loading ? (
        <Box sx={{ display: "flex", justifyContent: "center", p: 3 }}>
          <CircularProgress />
        </Box>
      ) : error ? (
        <Alert severity="error">{error}</Alert>
      ) : viewMode === "grid" ? (
        <Grid container spacing={3}>
          {actors
            .filter(
              (actor) =>
                searchTerm === "" ||
                String(actor.name || "")
                  .toLowerCase()
                  .includes(searchTerm.toLowerCase())
            )
            .map((actor) => (
              <Grid item xs={12} sm={6} md={4} lg={3} key={actor.id}>
                <Card>
                  <CardContent>
                    <Box sx={{ textAlign: "center", mb: 1 }}>
                      <Avatar
                        src={actor.imageUrl || actor.avatarUrl}
                        sx={{ width: 80, height: 80, mx: "auto" }}
                      />
                    </Box>
                    <Typography variant="h6" gutterBottom>
                      {actor.name}
                    </Typography>
                    {actor.dob && (
                      <Typography variant="caption" color="text.secondary">
                        DOB: {actor.dob}
                      </Typography>
                    )}
                  </CardContent>
                  <CardActions
                    sx={{ justifyContent: "space-between", px: 2, pb: 2 }}
                  >
                    <Tooltip title="View">
                      <IconButton
                        size="small"
                        color="primary"
                        onClick={() => setViewing(actor)}
                      >
                        <ViewIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Edit">
                      <IconButton
                        size="small"
                        color="secondary"
                        onClick={() => setEditing(actor)}
                      >
                        <EditIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete">
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => handleDeleted(actor)}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Tooltip>
                  </CardActions>
                </Card>
              </Grid>
            ))}
        </Grid>
      ) : (
        <>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Avatar</TableCell>
                  <TableCell>Name</TableCell>
                  <TableCell>DOB</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {actors
                  .filter(
                    (actor) =>
                      searchTerm === "" ||
                      String(actor.name || "")
                        .toLowerCase()
                        .includes(searchTerm.toLowerCase())
                  )
                  .map((actor) => (
                    <TableRow key={actor.id} hover>
                      <TableCell>
                        <Avatar
                          src={actor.imageUrl || actor.avatarUrl}
                          sx={{ width: 50, height: 50 }}
                        />
                      </TableCell>
                      <TableCell>{actor.name}</TableCell>
                      <TableCell>{actor.dob || ""}</TableCell>
                      <TableCell>
                        <Box sx={{ display: "flex", gap: 0.5 }}>
                          <Tooltip title="View">
                            <IconButton
                              size="small"
                              color="primary"
                              onClick={() => setViewing(actor)}
                            >
                              <ViewIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Edit">
                            <IconButton
                              size="small"
                              color="secondary"
                              onClick={() => setEditing(actor)}
                            >
                              <EditIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete">
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => handleDeleted(actor)}
                            >
                              <DeleteIcon />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
          </TableContainer>
          {totalPages > 1 && (
            <Box sx={{ display: "flex", justifyContent: "center", mt: 3 }}>
              <Pagination
                count={totalPages}
                page={page + 1}
                onChange={handlePageChange}
                color="primary"
                showFirstButton
                showLastButton
              />
            </Box>
          )}
        </>
      )}
      {viewMode === "grid" && totalPages > 1 && (
        <Box sx={{ display: "flex", justifyContent: "center", mt: 3 }}>
          <Pagination
            count={totalPages}
            page={page + 1}
            onChange={handlePageChange}
            color="primary"
            showFirstButton
            showLastButton
          />
        </Box>
      )}
      {totalPages > 0 && (
        <Box sx={{ display: "flex", justifyContent: "center", mt: 2 }}>
          <Typography variant="body2" color="text.secondary">
            Showing {actors.length} of {totalElements} actors
          </Typography>
        </Box>
      )}
    </Box>
  );
}

function ActorForm({ initialData, onSuccess, onCancel }) {
  const [formData, setFormData] = useState({
    name: initialData?.name || "",
    dob: initialData?.dob || "",
    description: initialData?.description || "",
  });
  const [file, setFile] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (field) => (e) =>
    setFormData((p) => ({ ...p, [field]: e.target.value }));
  const handleFile = (e) => setFile(e.target.files?.[0] || null);

  const onSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setError("");
    try {
      if (initialData?.id) {
        // If a new avatar selected, use multipart endpoint
        if (file) {
          await adminAPI.updateActorWithFormData(initialData.id, {
            ...formData,
            file,
          });
        } else {
          await adminAPI.updateActor(initialData.id, formData);
        }
      } else {
        await adminAPI.createActor({ ...formData, file });
      }
      onSuccess?.();
    } catch (err) {
      setError(err.message || "Failed to save actor");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Box component="form" onSubmit={onSubmit}>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            label="Name"
            value={formData.name}
            onChange={handleChange("name")}
            required
          />
        </Grid>
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            type="date"
            InputLabelProps={{ shrink: true }}
            label="Date of Birth"
            value={formData.dob}
            onChange={handleChange("dob")}
          />
        </Grid>
        <Grid item xs={12}>
          <TextField
            fullWidth
            multiline
            rows={3}
            label="Description"
            value={formData.description}
            onChange={handleChange("description")}
          />
        </Grid>

        <Grid item xs={12}>
          <input
            id="actor-file"
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={handleFile}
          />
          <label htmlFor="actor-file">
            <Button
              variant="outlined"
              component="span"
              startIcon={<CloudUploadIcon />}
            >
              {file
                ? file.name
                : initialData?.id
                ? "Change Avatar"
                : "Upload Avatar"}
            </Button>
          </label>
        </Grid>

        <Grid item xs={12}>
          <Box sx={{ display: "flex", gap: 2, justifyContent: "flex-end" }}>
            <Button onClick={onCancel} disabled={submitting}>
              Cancel
            </Button>
            <Button
              type="submit"
              variant="contained"
              disabled={submitting}
              startIcon={
                submitting ? <CircularProgress size={18} /> : <AddIcon />
              }
            >
              {initialData?.id ? "Update" : "Create"}
            </Button>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
}

function ActorViewForm({ actor, onClose }) {
  const [details, setDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    load();
  }, [actor.id]);

  const load = async () => {
    try {
      setLoading(true);
      const res = await adminAPI.getActorById(actor.id);
      setDetails(res?.data || res);
    } catch (err) {
      setError(err.message || "Failed to load actor");
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ display: "flex", justifyContent: "center", p: 3 }}>
            <CircularProgress />
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Alert severity="error">{error}</Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            mb: 3,
          }}
        >
          <Typography variant="h6">Actor Details: {details?.name}</Typography>
          <Button onClick={onClose} variant="outlined">
            Close
          </Button>
        </Box>

        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Box sx={{ textAlign: "center" }}>
              <Avatar
                src={details?.imageUrl}
                sx={{ width: 120, height: 120, mx: "auto" }}
              />
            </Box>
          </Grid>
          <Grid item xs={12} md={8}>
            <Typography variant="h5" gutterBottom>
              {details?.name}
            </Typography>
            {details?.dob && (
              <Typography variant="body2" color="text.secondary" gutterBottom>
                DOB: {details.dob}
              </Typography>
            )}
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Description
              </Typography>
              <Typography variant="body1">
                {details?.description || details?.biography}
              </Typography>
            </Box>
            {typeof details?.movieCount === "number" && (
              <Chip label={`Movies: ${details.movieCount}`} color="primary" />
            )}
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
}

// Country view component
function CountryViewForm({ country, onClose }) {
  const [countryDetails, setCountryDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    loadCountryDetails();
  }, [country.id]);

  const loadCountryDetails = async () => {
    try {
      setLoading(true);
      const response = await adminAPI.getCountryById(country.id);
      setCountryDetails(response?.data || response);
    } catch (err) {
      setError(err.message || "Failed to load country details");
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ display: "flex", justifyContent: "center", p: 3 }}>
            <CircularProgress />
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Alert severity="error">{error}</Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            mb: 3,
          }}
        >
          <Typography variant="h6">
            Country Details: {countryDetails?.name}
          </Typography>
          <Button onClick={onClose} variant="outlined">
            Close
          </Button>
        </Box>

        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Box sx={{ textAlign: "center" }}>
              <img
                src={countryDetails?.flagUrl || "/api/placeholder/200/150"}
                alt={`${countryDetails?.name} flag`}
                style={{
                  width: "200px",
                  height: "150px",
                  objectFit: "cover",
                  borderRadius: "8px",
                  border: "1px solid #ddd",
                }}
              />
            </Box>
          </Grid>

          <Grid item xs={12} md={8}>
            <Typography variant="h4" gutterBottom>
              {countryDetails?.name}
            </Typography>

            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Status
              </Typography>
              <Chip
                label={countryDetails?.isActive ? "Active" : "Inactive"}
                color={countryDetails?.isActive ? "success" : "default"}
                size="small"
              />
            </Box>

            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Country Information
              </Typography>
              <Box
                sx={{
                  p: 2,
                  bgcolor: "background.paper",
                  borderRadius: 1,
                  border: 1,
                  borderColor: "divider",
                }}
              >
                <Typography variant="body2" color="text.secondary">
                  <strong>Country ID:</strong> {countryDetails?.id}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  <strong>Created:</strong>{" "}
                  {new Date(countryDetails?.createdAt).toLocaleString()}
                </Typography>
                {countryDetails?.updatedAt && (
                  <Typography variant="body2" color="text.secondary">
                    <strong>Last Updated:</strong>{" "}
                    {new Date(countryDetails.updatedAt).toLocaleString()}
                  </Typography>
                )}
              </Box>
            </Box>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
}

// Movie view component
function MovieViewForm({ movie, onClose }) {
  const [movieDetails, setMovieDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [subtitleUploading, setSubtitleUploading] = useState(false);
  const [subtitleError, setSubtitleError] = useState("");
  const [subtitleLang, setSubtitleLang] = useState({
    code: "vi",
    name: "Vietnamese",
    isDefault: true,
  });

  useEffect(() => {
    loadMovieDetails();
  }, [movie.id]);

  const loadMovieDetails = async () => {
    try {
      setLoading(true);
      // Prefer public endpoint for richer payload
      let response = null;
      try {
        response = await adminAPI.getPublicMovieById(movie.id);
      } catch (_) {
        response = await adminAPI.getMovieById(movie.id);
      }
      setMovieDetails(response?.data || response);
    } catch (err) {
      setError(err.message || "Failed to load movie details");
    } finally {
      setLoading(false);
    }
  };

  const handleSubtitleSelect = async (event) => {
    const file = event.target.files?.[0];
    if (!file || !movie?.id) return;
    setSubtitleError("");
    setSubtitleUploading(true);
    try {
      await adminAPI.uploadMovieSubtitle(movie.id, file, {
        languageCode: subtitleLang.code || "vi",
        languageName: subtitleLang.name || "Vietnamese",
        isDefault:
          typeof subtitleLang.isDefault === "boolean"
            ? subtitleLang.isDefault
            : true,
      });
      await loadMovieDetails();
    } catch (err) {
      setSubtitleError(err?.message || "Failed to upload subtitle");
    } finally {
      setSubtitleUploading(false);
      // clear file input
      event.target.value = "";
    }
  };

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ display: "flex", justifyContent: "center", p: 3 }}>
            <CircularProgress />
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Alert severity="error">{error}</Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            mb: 3,
          }}
        >
          <Typography variant="h6">
            Movie Details: {movieDetails?.title}
          </Typography>
          <Button onClick={onClose} variant="outlined">
            Close
          </Button>
        </Box>

        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <CardMedia
              component="img"
              height="400"
              image={
                movieDetails?.posterUrl ||
                movieDetails?.thumbnailUrl ||
                "/api/placeholder/300/400"
              }
              alt={movieDetails?.title}
              sx={{
                objectFit: "cover",
                borderRadius: 1,
                background: "linear-gradient(45deg, #f0f0f0 30%, #e0e0e0 90%)",
              }}
            />
          </Grid>

          <Grid item xs={12} md={8}>
            <Typography variant="h4" gutterBottom>
              {movieDetails?.title}
            </Typography>
            <Typography variant="h6" color="text.secondary" gutterBottom>
              {movieDetails?.year}
            </Typography>

            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Synopsis
              </Typography>
              <Typography variant="body1">{movieDetails?.synopsis}</Typography>
            </Box>

            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Genres
              </Typography>
              <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}>
                {(movieDetails?.genres || movieDetails?.categories || []).map(
                  (g, index) => (
                    <Chip
                      key={index}
                      label={g}
                      color="primary"
                      variant="outlined"
                    />
                  )
                )}
              </Box>
            </Box>

            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Cast & Crew
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    <strong>Actors:</strong>{" "}
                    {(movieDetails?.actors && movieDetails.actors.join(", ")) ||
                      (movieDetails?.actorDetails &&
                        movieDetails.actorDetails
                          .map((a) => a?.name)
                          .filter(Boolean)
                          .join(", ")) ||
                      ""}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    <strong>Directors:</strong>{" "}
                    {(movieDetails?.directors &&
                      movieDetails.directors.join(", ")) ||
                      movieDetails?.directorName ||
                      ""}
                  </Typography>
                </Grid>
              </Grid>
            </Box>

            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Movie Information
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    <strong>Country:</strong> {movieDetails?.country}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    <strong>Language:</strong> {movieDetails?.language}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    <strong>Age Rating:</strong> {movieDetails?.ageRating}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    <strong>IMDB Rating:</strong>{" "}
                    {movieDetails?.imdbRating || "N/A"}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    <strong>Release Date:</strong>{" "}
                    {movieDetails?.releaseDate
                      ? new Date(movieDetails.releaseDate).toLocaleDateString()
                      : "N/A"}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    <strong>Max Download Quality:</strong>{" "}
                    {movieDetails?.maxDownloadQuality || "N/A"}
                  </Typography>
                </Grid>
                <Grid item xs={12}>
                  <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
                    <Chip
                      label={
                        movieDetails?.isAvailable ? "Available" : "Unavailable"
                      }
                      color={movieDetails?.isAvailable ? "success" : "default"}
                      size="small"
                    />
                    {movieDetails?.isFeatured && (
                      <Chip label="Featured" color="primary" size="small" />
                    )}
                    {movieDetails?.isTrending && (
                      <Chip label="Trending" color="secondary" size="small" />
                    )}
                    {movieDetails?.downloadEnabled && (
                      <Chip
                        label="Download Enabled"
                        color="info"
                        size="small"
                      />
                    )}
                  </Box>
                </Grid>
              </Grid>
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Typography variant="subtitle2" gutterBottom>
              Media Files
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={4}>
                <Box
                  sx={{
                    textAlign: "center",
                    p: 2,
                    border: 1,
                    borderColor: "divider",
                    borderRadius: 1,
                    bgcolor: "background.paper",
                  }}
                >
                  <Typography variant="h6" color="primary">
                    {movieDetails?.posterUrl ? "✓" : "✗"}
                  </Typography>
                  <Typography variant="caption">Poster</Typography>
                </Box>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Box
                  sx={{
                    textAlign: "center",
                    p: 2,
                    border: 1,
                    borderColor: "divider",
                    borderRadius: 1,
                    bgcolor: "background.paper",
                  }}
                >
                  <Typography variant="h6" color="secondary">
                    {movieDetails?.trailerUrl ? "✓" : "✗"}
                  </Typography>
                  <Typography variant="caption">Trailer</Typography>
                </Box>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Box
                  sx={{
                    textAlign: "center",
                    p: 2,
                    border: 1,
                    borderColor: "divider",
                    borderRadius: 1,
                    bgcolor: "background.paper",
                  }}
                >
                  <Typography variant="h6" color="success.main">
                    {Boolean(
                      movieDetails?.subtitleUrl ||
                        movieDetails?.subtitle ||
                        movieDetails?.subtitleFile ||
                        movieDetails?.subtitlePath ||
                        (Array.isArray(movieDetails?.subtitles) &&
                          movieDetails.subtitles.length > 0)
                    )
                      ? "✓"
                      : "✗"}
                  </Typography>
                  <Typography variant="caption">Subtitle</Typography>
                </Box>
              </Grid>
            </Grid>
          </Grid>

          <Grid item xs={12}>
            <Typography variant="subtitle2" gutterBottom>
              Movie Statistics
            </Typography>
            <Box
              sx={{
                p: 2,
                bgcolor: "background.paper",
                borderRadius: 1,
                border: 1,
                borderColor: "divider",
              }}
            >
              <Typography variant="body2" color="text.secondary">
                <strong>Created:</strong>{" "}
                {new Date(movieDetails?.createdAt).toLocaleString()}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                <strong>Movie ID:</strong> {movieDetails?.id}
              </Typography>
              {movieDetails?.updatedAt && (
                <Typography variant="body2" color="text.secondary">
                  <strong>Last Updated:</strong>{" "}
                  {new Date(movieDetails.updatedAt).toLocaleString()}
                </Typography>
              )}
            </Box>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
}

// Country creation form component
function CountryCreationForm({
  onSuccess,
  onCancel,
  initialData = null,
  isEdit = false,
}) {
  const [formData, setFormData] = useState({
    name: initialData?.name || "",
    isActive: initialData?.isActive ?? true,
  });

  const [flagFile, setFlagFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleInputChange = (field) => (event) => {
    const value =
      event.target.type === "checkbox"
        ? event.target.checked
        : event.target.value;
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setFlagFile(file);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");

    try {
      const countryData = {
        ...formData,
        flag: flagFile,
      };

      if (isEdit && initialData?.id) {
        await adminAPI.updateCountry(initialData.id, countryData);
      } else {
        await adminAPI.createCountry(countryData);
      }
      onSuccess?.();
    } catch (err) {
      setError(
        err.message || `Failed to ${isEdit ? "update" : "create"} country`
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          {isEdit ? "Edit Country" : "Create New Country"}
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <Box component="form" onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Country Name"
                value={formData.name}
                onChange={handleInputChange("name")}
                required
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={formData.isActive}
                    onChange={handleInputChange("isActive")}
                  />
                }
                label="Active"
              />
            </Grid>

            <Grid item xs={12}>
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Country Flag
                </Typography>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  style={{ display: "none" }}
                  id="flag-upload"
                />
                <label htmlFor="flag-upload">
                  <Button
                    variant="outlined"
                    component="span"
                    startIcon={<CloudUploadIcon />}
                    sx={{ mr: 2 }}
                  >
                    {flagFile ? flagFile.name : "Upload Flag Image"}
                  </Button>
                </label>
                {flagFile && (
                  <Typography variant="caption" color="text.secondary">
                    Selected: {flagFile.name}
                  </Typography>
                )}
              </Box>
            </Grid>

            <Grid item xs={12}>
              <Box sx={{ display: "flex", gap: 2, justifyContent: "flex-end" }}>
                <Button onClick={onCancel} disabled={loading}>
                  Cancel
                </Button>
                <Button
                  type="submit"
                  variant="contained"
                  disabled={loading}
                  startIcon={
                    loading ? <CircularProgress size={20} /> : <AddIcon />
                  }
                >
                  {loading
                    ? isEdit
                      ? "Updating..."
                      : "Creating..."
                    : isEdit
                    ? "Update Country"
                    : "Create Country"}
                </Button>
              </Box>
            </Grid>
          </Grid>
        </Box>
      </CardContent>
    </Card>
  );
}

// Movie creation form component
function MovieCreationForm({
  onSuccess,
  onCancel,
  initialData = null,
  isEdit = false,
}) {
  const [formData, setFormData] = useState({
    title: initialData?.title || "",
    synopsis: initialData?.synopsis || "",
    year: initialData?.year || new Date().getFullYear(),
    categories: initialData?.categories?.join(",") || "",
    actors: initialData?.actors?.join(",") || "",
    directors: initialData?.directors?.join(",") || "",
    country: initialData?.country || "",
    language: initialData?.language || "",
    ageRating: initialData?.ageRating || "",
    isAvailable: initialData?.isAvailable ?? true,
    imdbRating: initialData?.imdbRating || "",
    isFeatured: initialData?.isFeatured ?? false,
    isTrending: initialData?.isTrending ?? false,
    releaseDate: initialData?.releaseDate || "",
    downloadEnabled: initialData?.downloadEnabled ?? true,
    maxDownloadQuality: initialData?.maxDownloadQuality || "1080p",
  });

  const [files, setFiles] = useState({
    poster: null,
    thumbnail: null,
    video: null,
    trailer: null,
    subtitle: null,
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [uploadProgress, setUploadProgress] = useState({
    video: 0,
    trailer: 0,
    uploading: false,
    message: "",
  });

  // TMDB integration state
  const [showTmdbSearch, setShowTmdbSearch] = useState(false);

  // Lookup options for autocomplete fields
  const [actorOptions, setActorOptions] = useState([]);
  const [directorOptions, setDirectorOptions] = useState([]);
  const [countryOptions, setCountryOptions] = useState([]);
  const [categoryOptions, setCategoryOptions] = useState([]);

  useEffect(() => {
    const loadLookups = async () => {
      try {
        // Actors
        try {
          const a = await adminAPI.getActors(0, 50);
          const aList = Array.isArray(a) ? a : a?.content || a?.data || [];
          setActorOptions(aList.map((x) => x?.name || x).filter(Boolean));
        } catch (_) {
          setActorOptions([]);
        }

        // Directors
        try {
          let d = null;
          try {
            d = await adminAPI.getAllDirectors();
          } catch {
            d = await adminAPI.getDirectors(0, 50);
          }
          const dList = Array.isArray(d) ? d : d?.content || d?.data || [];
          setDirectorOptions(dList.map((x) => x?.name || x).filter(Boolean));
        } catch (_) {
          setDirectorOptions([]);
        }

        // Countries
        try {
          const c = await adminAPI.getCountries();
          const cList = Array.isArray(c) ? c : c?.content || c?.data || [];
          setCountryOptions(cList.map((x) => x?.name || x).filter(Boolean));
        } catch (_) {
          setCountryOptions([]);
        }

        // Categories
        try {
          const cat = await adminAPI.getCategories(0, 100);
          const catList = Array.isArray(cat)
            ? cat
            : cat?.content || cat?.data || [];
          setCategoryOptions(
            catList.map((x) => x?.name || x?.displayName || x).filter(Boolean)
          );
        } catch (_) {
          setCategoryOptions([]);
        }
      } catch (_) {
        // ignore; keep empty options
      }
    };
    loadLookups();
  }, []);

  // When editing, fetch full movie details from public API so fields are prefilled
  useEffect(() => {
    const loadEditDetails = async () => {
      if (!isEdit || !initialData?.id) return;
      try {
        let res = null;
        try {
          res = await adminAPI.getPublicMovieById(initialData.id);
        } catch (_) {
          res = await adminAPI.getMovieById(initialData.id);
        }
        const data = res?.data || res || {};
        const categories = (data.categories || data.genres || []).filter(
          Boolean
        );
        const actors =
          data.actors && Array.isArray(data.actors)
            ? data.actors
            : (data.actorDetails || []).map((a) => a?.name).filter(Boolean);
        const directors =
          data.directors && Array.isArray(data.directors)
            ? data.directors
            : [data.directorName].filter(Boolean);
        setFormData((prev) => ({
          ...prev,
          title: data.title ?? prev.title,
          synopsis: data.synopsis ?? prev.synopsis,
          year: data.year ?? prev.year,
          categories: categories.join(","),
          actors: actors.join(","),
          directors: directors.join(","),
          country: data.country ?? prev.country,
          language: data.language ?? prev.language,
          ageRating: data.ageRating ?? prev.ageRating,
          imdbRating: data.imdbRating ?? prev.imdbRating,
          releaseDate: data.releaseDate ?? prev.releaseDate,
          maxDownloadQuality:
            data.maxDownloadQuality ?? prev.maxDownloadQuality,
          isAvailable: data.isAvailable ?? prev.isAvailable,
          isFeatured: data.isFeatured ?? prev.isFeatured,
          isTrending: data.isTrending ?? prev.isTrending,
          downloadEnabled: data.downloadEnabled ?? prev.downloadEnabled,
        }));
      } catch (_) {
        // ignore, fallback to existing minimal data
      }
    };
    loadEditDetails();
  }, [isEdit, initialData?.id]);

  const handleInputChange = (field) => (event) => {
    const value =
      event.target.type === "checkbox"
        ? event.target.checked
        : event.target.value;
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleFileChange = (field) => (event) => {
    const file = event.target.files[0];
    setFiles((prev) => ({ ...prev, [field]: file }));
  };

  // Handle TMDB movie selection
  const handleTmdbMovieSelect = (tmdbMovieData) => {
    // Update form data with TMDB data
    setFormData((prev) => ({
      ...prev,
      title: tmdbMovieData.title || prev.title,
      synopsis: tmdbMovieData.synopsis || prev.synopsis,
      year: tmdbMovieData.year || prev.year,
      categories: tmdbMovieData.categories || prev.categories,
      actors: tmdbMovieData.actors || prev.actors,
      directors: tmdbMovieData.directors || prev.directors,
      country: tmdbMovieData.country || prev.country,
      language: tmdbMovieData.language || prev.language,
      ageRating: tmdbMovieData.ageRating || prev.ageRating,
      imdbRating: tmdbMovieData.imdbRating || prev.imdbRating,
      isFeatured: tmdbMovieData.isFeatured ?? prev.isFeatured,
      isTrending: tmdbMovieData.isTrending ?? prev.isTrending,
      releaseDate: tmdbMovieData.releaseDate || prev.releaseDate,
      trailerUrl: tmdbMovieData.trailerUrl || prev.trailerUrl,
    }));

    // Update files with downloaded images
    setFiles((prev) => ({
      ...prev,
      poster: tmdbMovieData.poster || prev.poster,
      thumbnail: tmdbMovieData.thumbnail || prev.thumbnail,
    }));

    setShowTmdbSearch(false);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setUploadProgress({ video: 0, trailer: 0, uploading: false, message: "" });

    try {
      const movieData = {
        ...formData,
        ...files,
      };

      if (isEdit && initialData?.id) {
        // Show progress for large file uploads
        if (files.video || files.trailer) {
          setUploadProgress((prev) => ({
            ...prev,
            uploading: true,
            message: "Đang lưu thông tin phim...",
          }));
        }

        // Use FormData for movie update to support file uploads
        // Video and trailer will be uploaded separately for better performance
        await adminAPI.updateMovieWithFormData(initialData.id, movieData);

        if (files.video || files.trailer) {
          setUploadProgress((prev) => ({
            ...prev,
            message: "Hoàn tất! Video đã được upload thành công.",
            video: 100,
            trailer: 100,
          }));
        }
      } else {
        await adminAPI.createMovie(movieData);
      }
      onSuccess?.();
    } catch (err) {
      setError(
        err.message || `Failed to ${isEdit ? "update" : "create"} movie`
      );
    } finally {
      setLoading(false);
      setTimeout(() => {
        setUploadProgress({
          video: 0,
          trailer: 0,
          uploading: false,
          message: "",
        });
      }, 2000);
    }
  };

  return (
    <Card>
      <CardContent>
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            mb: 2,
          }}
        >
          <Typography variant="h6">
            {isEdit ? "Edit Movie" : "Create New Movie"}
          </Typography>
          {!isEdit && (
            <Button
              variant="outlined"
              startIcon={<MovieIcon />}
              onClick={() => setShowTmdbSearch(true)}
              color="primary"
            >
              Import from TMDB
            </Button>
          )}
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {uploadProgress.uploading && uploadProgress.message && (
          <Alert severity="info" sx={{ mb: 2 }}>
            {uploadProgress.message}
          </Alert>
        )}

        <Box component="form" onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Title"
                value={formData.title}
                onChange={handleInputChange("title")}
                required
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Year"
                type="number"
                value={formData.year}
                onChange={handleInputChange("year")}
                required
              />
            </Grid>

            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Synopsis"
                multiline
                rows={4}
                value={formData.synopsis}
                onChange={handleInputChange("synopsis")}
                required
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <Autocomplete
                multiple
                options={
                  actorOptions && Array.isArray(categoryOptions)
                    ? categoryOptions
                    : []
                }
                filterSelectedOptions
                value={(formData.categories || "").split(",").filter(Boolean)}
                onChange={(_, value) =>
                  setFormData((p) => ({
                    ...p,
                    categories: (value || []).join(","),
                  }))
                }
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label="Categories"
                    placeholder="Start typing..."
                  />
                )}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <Autocomplete
                multiple
                options={Array.isArray(actorOptions) ? actorOptions : []}
                filterSelectedOptions
                value={(formData.actors || "").split(",").filter(Boolean)}
                onChange={(_, value) =>
                  setFormData((p) => ({
                    ...p,
                    actors: (value || []).join(","),
                  }))
                }
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label="Actors"
                    placeholder="Start typing..."
                  />
                )}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <Autocomplete
                multiple
                options={Array.isArray(directorOptions) ? directorOptions : []}
                filterSelectedOptions
                value={(formData.directors || "").split(",").filter(Boolean)}
                onChange={(_, value) =>
                  setFormData((p) => ({
                    ...p,
                    directors: (value || []).join(","),
                  }))
                }
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label="Directors"
                    placeholder="Start typing..."
                  />
                )}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <Autocomplete
                options={Array.isArray(countryOptions) ? countryOptions : []}
                value={formData.country || null}
                onChange={(_, value) =>
                  setFormData((p) => ({ ...p, country: value || "" }))
                }
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label="Country"
                    placeholder="Start typing..."
                  />
                )}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Language"
                value={formData.language}
                onChange={handleInputChange("language")}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Age Rating</InputLabel>
                <Select
                  value={formData.ageRating}
                  onChange={handleInputChange("ageRating")}
                  label="Age Rating"
                >
                  <MenuItem value="G">G</MenuItem>
                  <MenuItem value="PG">PG</MenuItem>
                  <MenuItem value="PG-13">PG-13</MenuItem>
                  <MenuItem value="R">R</MenuItem>
                  <MenuItem value="NC-17">NC-17</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={4}>
              <input
                accept="image/*"
                style={{ display: "none" }}
                id="poster-upload"
                type="file"
                onChange={handleFileChange("poster")}
              />
              <label htmlFor="poster-upload">
                <Button variant="outlined" component="span" fullWidth>
                  {files.poster ? files.poster.name : "Upload Poster"}
                </Button>
              </label>
            </Grid>

            <Grid item xs={12} md={4}>
              <input
                accept="image/*"
                style={{ display: "none" }}
                id="thumbnail-upload"
                type="file"
                onChange={handleFileChange("thumbnail")}
              />
              <label htmlFor="thumbnail-upload">
                <Button variant="outlined" component="span" fullWidth>
                  {files.thumbnail ? files.thumbnail.name : "Upload Thumbnail"}
                </Button>
              </label>
            </Grid>

            <Grid item xs={12} md={4}>
              <input
                accept="video/*"
                style={{ display: "none" }}
                id="trailer-upload"
                type="file"
                onChange={handleFileChange("trailer")}
              />
              <label htmlFor="trailer-upload">
                <Button variant="outlined" component="span" fullWidth>
                  {files.trailer ? files.trailer.name : "Upload Trailer"}
                </Button>
              </label>
            </Grid>

            {isEdit && (
              <Grid item xs={12} md={4}>
                <input
                  accept="video/*"
                  style={{ display: "none" }}
                  id="video-upload"
                  type="file"
                  onChange={handleFileChange("video")}
                />
                <label htmlFor="video-upload">
                  <Button variant="outlined" component="span" fullWidth>
                    {files.video ? files.video.name : "Upload Video"}
                  </Button>
                </label>
              </Grid>
            )}

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="IMDB Rating"
                type="number"
                step="0.1"
                value={formData.imdbRating}
                onChange={handleInputChange("imdbRating")}
                placeholder="8.5"
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Release Date"
                type="date"
                value={formData.releaseDate}
                onChange={handleInputChange("releaseDate")}
                InputLabelProps={{ shrink: true }}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Max Download Quality</InputLabel>
                <Select
                  value={formData.maxDownloadQuality}
                  onChange={handleInputChange("maxDownloadQuality")}
                  label="Max Download Quality"
                >
                  <MenuItem value="480p">480p</MenuItem>
                  <MenuItem value="720p">720p</MenuItem>
                  <MenuItem value="1080p">1080p</MenuItem>
                  <MenuItem value="4K">4K</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={6}>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={formData.isAvailable}
                      onChange={handleInputChange("isAvailable")}
                    />
                  }
                  label="Available for streaming"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={formData.isFeatured}
                      onChange={handleInputChange("isFeatured")}
                    />
                  }
                  label="Featured movie"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={formData.isTrending}
                      onChange={handleInputChange("isTrending")}
                    />
                  }
                  label="Trending movie"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={formData.downloadEnabled}
                      onChange={handleInputChange("downloadEnabled")}
                    />
                  }
                  label="Download enabled"
                />
              </Box>
            </Grid>

            <Grid item xs={12}>
              <Box sx={{ display: "flex", gap: 2, justifyContent: "flex-end" }}>
                <Button onClick={onCancel} disabled={loading}>
                  Cancel
                </Button>
                <Button
                  type="submit"
                  variant="contained"
                  disabled={loading}
                  startIcon={
                    loading ? <CircularProgress size={20} /> : <AddIcon />
                  }
                >
                  {loading
                    ? isEdit
                      ? "Updating..."
                      : "Creating..."
                    : isEdit
                    ? "Update Movie"
                    : "Create Movie"}
                </Button>
              </Box>
            </Grid>
          </Grid>
        </Box>
      </CardContent>

      {/* TMDB Movie Search Modal */}
      <TmdbMovieSearch
        open={showTmdbSearch}
        onClose={() => setShowTmdbSearch(false)}
        onSelectMovie={handleTmdbMovieSelect}
      />
    </Card>
  );
}

// Movies management component
function MoviesManagement() {
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [viewMode, setViewMode] = useState("grid"); // "grid" or "table"
  const [editingMovie, setEditingMovie] = useState(null);
  const [viewingMovie, setViewingMovie] = useState(null);
  const [deleteDialog, setDeleteDialog] = useState({
    open: false,
    movie: null,
  });
  const [page, setPage] = useState(0);
  const [size] = useState(20);
  const [totalPages, setTotalPages] = useState(0);
  const [totalElements, setTotalElements] = useState(0);
  const [usePagination, setUsePagination] = useState(true); // Toggle pagination
  const [allMovies, setAllMovies] = useState([]); // Store all movies for client-side pagination
  const subtitleInputRef = useRef(null);
  const [subtitleUploadMovieId, setSubtitleUploadMovieId] = useState(null);
  const [subtitleUploadError, setSubtitleUploadError] = useState("");
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    loadMovies();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, usePagination]);

  const loadMovies = async () => {
    try {
      setLoading(true);
      setError("");
      let response;

      if (usePagination) {
        try {
          // Try with pagination first
          response = await adminAPI.getMovies(page, size);
        } catch (paginationError) {
          // If pagination fails, disable it and load all
          console.warn(
            "Pagination failed, loading all movies:",
            paginationError
          );
          setUsePagination(false);
          response = await adminAPI.getMovies();
        }
      } else {
        // Load all without pagination (only once)
        if (allMovies.length === 0) {
          response = await adminAPI.getMovies();
        } else {
          // Use cached data for client-side pagination
          const startIndex = page * size;
          const endIndex = startIndex + size;
          setMovies(allMovies.slice(startIndex, endIndex));
          setTotalPages(Math.ceil(allMovies.length / size));
          setTotalElements(allMovies.length);
          setLoading(false);
          return;
        }
      }

      // Handle paginated response
      if (response?.content) {
        const allMoviesList = response.content;
        if (usePagination) {
          setMovies(allMoviesList);
          setTotalPages(response.totalPages || 0);
          setTotalElements(response.totalElements || 0);
        } else {
          setAllMovies(allMoviesList);
          const startIndex = page * size;
          const endIndex = startIndex + size;
          setMovies(allMoviesList.slice(startIndex, endIndex));
          setTotalPages(Math.ceil(allMoviesList.length / size));
          setTotalElements(allMoviesList.length);
        }
      } else if (response?.data?.content) {
        const allMoviesList = response.data.content;
        if (usePagination) {
          setMovies(allMoviesList);
          setTotalPages(response.data.totalPages || 0);
          setTotalElements(response.data.totalElements || 0);
        } else {
          setAllMovies(allMoviesList);
          const startIndex = page * size;
          const endIndex = startIndex + size;
          setMovies(allMoviesList.slice(startIndex, endIndex));
          setTotalPages(Math.ceil(allMoviesList.length / size));
          setTotalElements(allMoviesList.length);
        }
      } else {
        const list = Array.isArray(response) ? response : response?.data || [];
        if (usePagination) {
          setMovies(list);
          setTotalPages(1);
          setTotalElements(list.length);
        } else {
          setAllMovies(list);
          const startIndex = page * size;
          const endIndex = startIndex + size;
          setMovies(list.slice(startIndex, endIndex));
          setTotalPages(Math.ceil(list.length / size));
          setTotalElements(list.length);
        }
      }
    } catch (err) {
      console.error("Error loading movies:", err);
      setError(err.message || "Failed to load movies");
      setMovies([]);
      setTotalPages(0);
      setTotalElements(0);
    } finally {
      setLoading(false);
    }
  };

  const triggerSubtitleUpload = (movie) => {
    setSubtitleUploadError("");
    setSubtitleUploadMovieId(movie?.id || null);
    if (subtitleInputRef.current) {
      subtitleInputRef.current.value = "";
      subtitleInputRef.current.click();
    }
  };

  const handleSubtitleUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file || !subtitleUploadMovieId) return;
    try {
      setSubtitleUploadError("");
      await adminAPI.uploadMovieSubtitle(subtitleUploadMovieId, file, {
        languageCode: "vi",
        languageName: "Vietnamese",
        isDefault: true,
      });
      await loadMovies();
    } catch (err) {
      setSubtitleUploadError(
        err?.message || "Upload subtitle failed. Please try again."
      );
    } finally {
      event.target.value = "";
      setSubtitleUploadMovieId(null);
    }
  };

  const handlePageChange = (event, newPage) => {
    setPage(newPage - 1); // MUI Pagination uses 1-based indexing
  };

  const handleCreateSuccess = () => {
    setShowCreateForm(false);
    setAllMovies([]); // Clear cache to reload
    setPage(0); // Reset to first page
    loadMovies();
  };

  const handleViewMovie = (movie) => {
    setViewingMovie(movie);
  };

  const handleEditMovie = (movie) => {
    setEditingMovie(movie);
  };

  const handleDeleteMovie = (movie) => {
    setDeleteDialog({ open: true, movie });
  };

  const handleToggleAvailability = async (movie) => {
    try {
      await adminAPI.toggleMovieAvailability(movie.id, !movie.isAvailable);
      // Update local state if using client-side pagination
      if (!usePagination && allMovies.length > 0) {
        const updatedMovies = allMovies.map((m) =>
          m.id === movie.id ? { ...m, isAvailable: !movie.isAvailable } : m
        );
        setAllMovies(updatedMovies);
        const startIndex = page * size;
        const endIndex = startIndex + size;
        setMovies(updatedMovies.slice(startIndex, endIndex));
      } else {
        loadMovies();
      }
    } catch (err) {
      console.error("Failed to toggle movie availability:", err);
    }
  };

  const confirmDeleteMovie = async () => {
    try {
      await adminAPI.deleteMovie(deleteDialog.movie.id);
      setDeleteDialog({ open: false, movie: null });
      setAllMovies([]); // Clear cache to reload
      loadMovies();
    } catch (err) {
      console.error("Failed to delete movie:", err);
    }
  };

  const handleEditSuccess = () => {
    setEditingMovie(null);
    setAllMovies([]); // Clear cache to reload
    loadMovies();
  };

  if (viewingMovie) {
    return (
      <MovieViewForm
        movie={viewingMovie}
        onClose={() => setViewingMovie(null)}
      />
    );
  }

  if (showCreateForm) {
    return (
      <MovieCreationForm
        onSuccess={handleCreateSuccess}
        onCancel={() => setShowCreateForm(false)}
      />
    );
  }

  if (editingMovie) {
    return (
      <MovieCreationForm
        initialData={editingMovie}
        onSuccess={handleEditSuccess}
        onCancel={() => setEditingMovie(null)}
        isEdit={true}
      />
    );
  }

  return (
    <Box>
      <input
        ref={subtitleInputRef}
        type="file"
        accept=".srt,.vtt,.ass,.sub"
        style={{ display: "none" }}
        onChange={handleSubtitleUpload}
      />
      {subtitleUploadError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {subtitleUploadError}
        </Alert>
      )}
      {/* Search */}
      <Box sx={{ mb: 3 }}>
        <TextField
          placeholder="Search movies by title..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          size="small"
          sx={{ minWidth: 300 }}
          fullWidth
        />
      </Box>
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          mb: 3,
        }}
      >
        <Typography
          variant="h5"
          sx={{ display: "flex", alignItems: "center", gap: 1 }}
        >
          <MovieIcon color="primary" />
          Movies Management
        </Typography>
        <Box sx={{ display: "flex", gap: 2, alignItems: "center" }}>
          <Button
            variant={viewMode === "grid" ? "contained" : "outlined"}
            size="small"
            onClick={() => setViewMode("grid")}
          >
            Grid
          </Button>
          <Button
            variant={viewMode === "table" ? "contained" : "outlined"}
            size="small"
            onClick={() => setViewMode("table")}
          >
            Table
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setShowCreateForm(true)}
          >
            Add Movie
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {loading ? (
        <Box sx={{ display: "flex", justifyContent: "center", p: 3 }}>
          <CircularProgress />
        </Box>
      ) : viewMode === "grid" ? (
        <Grid container spacing={3}>
          {movies
            .filter(
              (movie) =>
                searchTerm === "" ||
                String(movie.title || "")
                  .toLowerCase()
                  .includes(searchTerm.toLowerCase())
            )
            .map((movie) => (
              <Grid item xs={12} sm={6} md={4} lg={3} key={movie.id}>
                <Card
                  sx={{
                    height: "100%",
                    display: "flex",
                    flexDirection: "column",
                    transition:
                      "transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out",
                    "&:hover": {
                      transform: "translateY(-4px)",
                      boxShadow: (theme) => theme.shadows[8],
                    },
                  }}
                >
                  <CardMedia
                    component="img"
                    height="300"
                    image={
                      movie.posterUrl ||
                      movie.thumbnailUrl ||
                      "/api/placeholder/300/400"
                    }
                    alt={movie.title}
                    sx={{
                      objectFit: "cover",
                      background:
                        "linear-gradient(45deg, #f0f0f0 30%, #e0e0e0 90%)",
                    }}
                  />
                  <CardContent sx={{ flexGrow: 1, pb: 1 }}>
                    <Typography variant="h6" component="h2" gutterBottom noWrap>
                      {movie.title}
                    </Typography>
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      gutterBottom
                    >
                      {movie.year}
                    </Typography>
                    <Box sx={{ mb: 1 }}>
                      {movie.genres?.slice(0, 2).map((genre, index) => (
                        <Chip
                          key={index}
                          label={genre}
                          size="small"
                          sx={{ mr: 0.5, mb: 0.5 }}
                          color="primary"
                          variant="outlined"
                        />
                      ))}
                      {movie.genres?.length > 2 && (
                        <Chip
                          label={`+${movie.genres.length - 2}`}
                          size="small"
                          sx={{ mr: 0.5, mb: 0.5 }}
                          color="default"
                          variant="outlined"
                        />
                      )}
                    </Box>
                    <Box
                      sx={{
                        display: "flex",
                        gap: 1,
                        alignItems: "center",
                        mb: 1,
                      }}
                    >
                      <Chip
                        label={movie.isAvailable ? "Available" : "Unavailable"}
                        color={movie.isAvailable ? "success" : "default"}
                        size="small"
                      />
                      <FormControlLabel
                        control={
                          <Switch
                            checked={movie.isAvailable}
                            onChange={() => handleToggleAvailability(movie)}
                            size="small"
                          />
                        }
                        label=""
                      />
                    </Box>
                  </CardContent>
                  <CardActions
                    sx={{ justifyContent: "space-between", px: 2, pb: 2 }}
                  >
                    <Tooltip title="View Details">
                      <IconButton
                        size="small"
                        color="primary"
                        onClick={() => handleViewMovie(movie)}
                      >
                        <ViewIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Edit Movie">
                      <IconButton
                        size="small"
                        color="secondary"
                        onClick={() => handleEditMovie(movie)}
                      >
                        <EditIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Upload Subtitle">
                      <IconButton
                        size="small"
                        color="info"
                        onClick={() => triggerSubtitleUpload(movie)}
                      >
                        <ClosedCaptionIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete Movie">
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => handleDeleteMovie(movie)}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Tooltip>
                  </CardActions>
                </Card>
              </Grid>
            ))}
        </Grid>
      ) : (
        <>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Poster</TableCell>
                  <TableCell>Title</TableCell>
                  <TableCell>Year</TableCell>
                  <TableCell>Categories</TableCell>
                  <TableCell>Available</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {movies
                  .filter(
                    (movie) =>
                      searchTerm === "" ||
                      String(movie.title || "")
                        .toLowerCase()
                        .includes(searchTerm.toLowerCase())
                  )
                  .map((movie) => (
                    <TableRow key={movie.id} hover>
                      <TableCell>
                        <Avatar
                          src={movie.posterUrl || movie.thumbnailUrl}
                          alt={movie.title}
                          variant="rounded"
                          sx={{ width: 60, height: 80 }}
                        />
                      </TableCell>
                      <TableCell>
                        <Typography variant="subtitle2" fontWeight="bold">
                          {movie.title}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {movie.synopsis?.substring(0, 50)}...
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">{movie.year}</Typography>
                      </TableCell>
                      <TableCell>
                        <Box
                          sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}
                        >
                          {movie.genres?.slice(0, 2).map((genre, index) => (
                            <Chip
                              key={index}
                              label={genre}
                              size="small"
                              color="primary"
                              variant="outlined"
                            />
                          ))}
                          {movie.genres?.length > 2 && (
                            <Chip
                              label={`+${movie.genres.length - 2}`}
                              size="small"
                              color="default"
                              variant="outlined"
                            />
                          )}
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Box
                          sx={{ display: "flex", gap: 1, alignItems: "center" }}
                        >
                          <Chip
                            label={
                              movie.isAvailable ? "Available" : "Unavailable"
                            }
                            color={movie.isAvailable ? "success" : "default"}
                            size="small"
                          />
                          <Switch
                            checked={movie.isAvailable}
                            onChange={() => handleToggleAvailability(movie)}
                            size="small"
                          />
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: "flex", gap: 0.5 }}>
                          <Tooltip title="View Details">
                            <IconButton
                              size="small"
                              color="primary"
                              onClick={() => handleViewMovie(movie)}
                            >
                              <ViewIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Edit Movie">
                            <IconButton
                              size="small"
                              color="secondary"
                              onClick={() => handleEditMovie(movie)}
                            >
                              <EditIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Upload Subtitle">
                            <IconButton
                              size="small"
                              color="info"
                              onClick={() => triggerSubtitleUpload(movie)}
                            >
                              <ClosedCaptionIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete Movie">
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => handleDeleteMovie(movie)}
                            >
                              <DeleteIcon />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
          </TableContainer>
          {totalPages > 1 && (
            <Box sx={{ display: "flex", justifyContent: "center", mt: 3 }}>
              <Pagination
                count={totalPages}
                page={page + 1}
                onChange={handlePageChange}
                color="primary"
                showFirstButton
                showLastButton
              />
            </Box>
          )}
        </>
      )}
      {viewMode === "grid" && totalPages > 1 && (
        <Box sx={{ display: "flex", justifyContent: "center", mt: 3 }}>
          <Pagination
            count={totalPages}
            page={page + 1}
            onChange={handlePageChange}
            color="primary"
            showFirstButton
            showLastButton
          />
        </Box>
      )}
      {totalPages > 0 && (
        <Box sx={{ display: "flex", justifyContent: "center", mt: 2 }}>
          <Typography variant="body2" color="text.secondary">
            Showing {movies.length} of {totalElements} movies
          </Typography>
        </Box>
      )}

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialog.open}
        onClose={() => setDeleteDialog({ open: false, movie: null })}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete &quot;{deleteDialog.movie?.title}
            &quot;? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialog({ open: false, movie: null })}>
            Cancel
          </Button>
          <Button
            onClick={confirmDeleteMovie}
            color="error"
            variant="contained"
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

// User view component
function UserViewForm({ user, onClose }) {
  const [userDetails, setUserDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    loadUserDetails();
  }, [user.id]);

  const loadUserDetails = async () => {
    try {
      setLoading(true);
      const response = await adminAPI.getUserById(user.id);
      setUserDetails(response);
    } catch (err) {
      setError(err.message || "Failed to load user details");
    } finally {
      setLoading(false);
    }
  };

  const getInitials = (name) => {
    if (!name) return "U";
    return name
      .split(" ")
      .map((word) => word.charAt(0))
      .join("")
      .toUpperCase()
      .slice(0, 2);
  };

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ display: "flex", justifyContent: "center", p: 3 }}>
            <CircularProgress />
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Alert severity="error">{error}</Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            mb: 3,
          }}
        >
          <Typography variant="h6">
            User Details: {userDetails?.username}
          </Typography>
          <Button onClick={onClose} variant="outlined">
            Close
          </Button>
        </Box>

        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
              <Avatar
                src={userDetails?.avatarUrl}
                sx={{
                  width: 80,
                  height: 80,
                  background: userDetails?.avatarUrl
                    ? "transparent"
                    : "linear-gradient(135deg, #FFD700 0%, #FFA500 100%)",
                  color: userDetails?.avatarUrl ? "transparent" : "#000",
                  fontWeight: 700,
                  fontSize: "1.5rem",
                }}
              >
                {!userDetails?.avatarUrl &&
                  getInitials(userDetails?.fullName || userDetails?.username)}
              </Avatar>
              <Box>
                <Typography variant="h5" gutterBottom>
                  {userDetails?.fullName || userDetails?.username}
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  @{userDetails?.username}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {userDetails?.email}
                </Typography>
              </Box>
            </Box>
          </Grid>

          <Grid item xs={12} md={6}>
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Account Status
              </Typography>
              <Chip
                label={userDetails?.enabled ? "Active" : "Inactive"}
                color={userDetails?.enabled ? "success" : "default"}
                sx={{ mb: 1 }}
              />
            </Box>

            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Roles
              </Typography>
              <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}>
                {userDetails?.roles?.map((role, index) => (
                  <Chip
                    key={index}
                    label={role}
                    color={role === "ADMIN" ? "error" : "primary"}
                    variant="outlined"
                  />
                ))}
              </Box>
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Typography variant="subtitle2" gutterBottom>
              Account Statistics
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={6} sm={3}>
                <Box
                  sx={{
                    textAlign: "center",
                    p: 2,
                    border: 1,
                    borderColor: "divider",
                    borderRadius: 1,
                    bgcolor: "background.paper",
                  }}
                >
                  <Typography variant="h6" color="primary">
                    {userDetails?.totalComments || 0}
                  </Typography>
                  <Typography variant="caption">Comments</Typography>
                </Box>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Box
                  sx={{
                    textAlign: "center",
                    p: 2,
                    border: 1,
                    borderColor: "divider",
                    borderRadius: 1,
                    bgcolor: "background.paper",
                  }}
                >
                  <Typography variant="h6" color="secondary">
                    {userDetails?.totalReports || 0}
                  </Typography>
                  <Typography variant="caption">Reports</Typography>
                </Box>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Box
                  sx={{
                    textAlign: "center",
                    p: 2,
                    border: 1,
                    borderColor: "divider",
                    borderRadius: 1,
                    bgcolor: "background.paper",
                  }}
                >
                  <Typography variant="h6" color="success.main">
                    {userDetails?.totalMovies || 0}
                  </Typography>
                  <Typography variant="caption">Movies</Typography>
                </Box>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Box
                  sx={{
                    textAlign: "center",
                    p: 2,
                    border: 1,
                    borderColor: "divider",
                    borderRadius: 1,
                    bgcolor: "background.paper",
                  }}
                >
                  <Typography variant="h6" color="info.main">
                    {userDetails?.lastLoginAt
                      ? new Date(userDetails.lastLoginAt).toLocaleDateString()
                      : "Never"}
                  </Typography>
                  <Typography variant="caption">Last Login</Typography>
                </Box>
              </Grid>
            </Grid>
          </Grid>

          <Grid item xs={12}>
            <Typography variant="subtitle2" gutterBottom>
              Account Information
            </Typography>
            <Box
              sx={{
                p: 2,
                bgcolor: "background.paper",
                borderRadius: 1,
                border: 1,
                borderColor: "divider",
              }}
            >
              <Typography variant="body2" color="text.secondary">
                <strong>Created:</strong>{" "}
                {new Date(userDetails?.createdAt).toLocaleString()}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                <strong>User ID:</strong> {userDetails?.id}
              </Typography>
              {userDetails?.lastLoginAt && (
                <Typography variant="body2" color="text.secondary">
                  <strong>Last Login:</strong>{" "}
                  {new Date(userDetails.lastLoginAt).toLocaleString()}
                </Typography>
              )}
            </Box>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
}

// User edit form component (simplified - only status and roles)
function UserEditForm({ user, onSuccess, onCancel }) {
  const [formData, setFormData] = useState({
    enabled: user?.enabled ?? true,
    roles: user?.roles || ["USER"], // Default to USER role if no roles exist
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const availableRoles = ["USER", "ADMIN"];

  const handleInputChange = (field) => (event) => {
    const value =
      event.target.type === "checkbox"
        ? event.target.checked
        : event.target.value;
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleRoleChange = (event) => {
    const value = event.target.value;
    setFormData((prev) => ({
      ...prev,
      roles: [value], // Wrap single value in array to maintain API compatibility
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");

    try {
      // Update user status if changed
      if (formData.enabled !== user.enabled) {
        await adminAPI.updateUserStatus(user.id, formData.enabled);
      }

      // Update user roles if changed
      if (
        JSON.stringify(formData.roles.sort()) !==
        JSON.stringify(user.roles?.sort() || [])
      ) {
        await adminAPI.updateUserRoles(user.id, formData.roles);
      }

      onSuccess?.();
    } catch (err) {
      setError(err.message || "Failed to update user");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Edit User: {user?.username}
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <Box component="form" onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Role</InputLabel>
                <Select
                  value={formData.roles[0] || ""}
                  onChange={handleRoleChange}
                  label="Role"
                >
                  {availableRoles.map((role) => (
                    <MenuItem key={role} value={role}>
                      {role}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={formData.enabled}
                    onChange={handleInputChange("enabled")}
                  />
                }
                label="User Enabled"
              />
            </Grid>

            <Grid item xs={12}>
              <Box sx={{ display: "flex", gap: 2, justifyContent: "flex-end" }}>
                <Button onClick={onCancel} disabled={loading}>
                  Cancel
                </Button>
                <Button
                  type="submit"
                  variant="contained"
                  disabled={loading}
                  startIcon={
                    loading ? <CircularProgress size={20} /> : <EditIcon />
                  }
                >
                  {loading ? "Updating..." : "Update User"}
                </Button>
              </Box>
            </Grid>
          </Grid>
        </Box>
      </CardContent>
    </Card>
  );
}

// Bulk operations dialog
function BulkOperationsDialog({
  open,
  onClose,
  onSuccess,
  selectedUsers = [],
}) {
  const [reason, setReason] = useState("Policy violation");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleBulkDisable = async () => {
    if (selectedUsers.length === 0) return;

    setLoading(true);
    setError("");

    try {
      const userIds = selectedUsers.map((user) => user.id);
      await adminAPI.bulkDisableUsers(userIds, reason);
      onSuccess?.();
      onClose();
    } catch (err) {
      setError(err.message || "Failed to disable users");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Bulk Disable Users</DialogTitle>
      <DialogContent>
        <Typography variant="body1" gutterBottom>
          You are about to disable {selectedUsers.length} user(s). This action
          will prevent them from accessing the platform.
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <TextField
          fullWidth
          label="Reason for disabling"
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          multiline
          rows={3}
          sx={{ mt: 2 }}
        />

        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          Selected users:{" "}
          {selectedUsers.map((user) => user.username).join(", ")}
        </Typography>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={loading}>
          Cancel
        </Button>
        <Button
          onClick={handleBulkDisable}
          color="error"
          variant="contained"
          disabled={loading || selectedUsers.length === 0}
          startIcon={loading ? <CircularProgress size={20} /> : <DeleteIcon />}
        >
          {loading ? "Disabling..." : "Disable Users"}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

// Countries management component
function CountriesManagement() {
  const [countries, setCountries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [viewMode, setViewMode] = useState("grid"); // "grid" or "table"
  const [editingCountry, setEditingCountry] = useState(null);
  const [viewingCountry, setViewingCountry] = useState(null);
  const [deleteDialog, setDeleteDialog] = useState({
    open: false,
    country: null,
  });
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    loadCountries();
  }, []);

  const loadCountries = async () => {
    try {
      setLoading(true);
      const response = await adminAPI.getCountries();
      // Support various API response shapes: array, {content}, {data}
      const list = Array.isArray(response)
        ? response
        : response?.content || response?.data || [];
      setCountries(list);
    } catch (err) {
      setError(err.message || "Failed to load countries");
    } finally {
      setLoading(false);
    }
  };

  const handleCreateSuccess = () => {
    setShowCreateForm(false);
    setSuccess("Country created successfully");
    loadCountries();
  };

  const handleViewCountry = (country) => {
    setViewingCountry(country);
  };

  const handleEditCountry = (country) => {
    setEditingCountry(country);
  };

  const handleDeleteCountry = (country) => {
    setDeleteDialog({ open: true, country });
  };

  const handleToggleActivation = async (country) => {
    try {
      if (country.isActive) {
        await adminAPI.deactivateCountry(country.id);
      } else {
        await adminAPI.activateCountry(country.id);
      }
      loadCountries();
    } catch (err) {
      console.error("Failed to toggle country activation:", err);
    }
  };

  const confirmDeleteCountry = async () => {
    try {
      await adminAPI.deleteCountry(deleteDialog.country.id);
      setDeleteDialog({ open: false, country: null });
      loadCountries();
    } catch (err) {
      console.error("Failed to delete country:", err);
    }
  };

  const handleEditSuccess = () => {
    setEditingCountry(null);
    loadCountries();
  };

  if (viewingCountry) {
    return (
      <CountryViewForm
        country={viewingCountry}
        onClose={() => setViewingCountry(null)}
      />
    );
  }

  if (showCreateForm) {
    return (
      <CountryCreationForm
        onSuccess={handleCreateSuccess}
        onCancel={() => setShowCreateForm(false)}
      />
    );
  }

  if (editingCountry) {
    return (
      <CountryCreationForm
        initialData={editingCountry}
        onSuccess={handleEditSuccess}
        onCancel={() => setEditingCountry(null)}
        isEdit={true}
      />
    );
  }

  return (
    <Box>
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          mb: 3,
        }}
      >
        <Typography variant="h5">Countries Management</Typography>
        <Box sx={{ display: "flex", gap: 2 }}>
          <Button
            variant="outlined"
            onClick={() => setViewMode(viewMode === "grid" ? "table" : "grid")}
            startIcon={
              viewMode === "grid" ? <ViewListIcon /> : <ViewModuleIcon />
            }
          >
            {viewMode === "grid" ? "Table View" : "Grid View"}
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setShowCreateForm(true)}
          >
            Add Country
          </Button>
        </Box>
      </Box>

      {success && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess("")}>
          {success}
        </Alert>
      )}
      {/* Search */}
      <Box sx={{ mb: 3 }}>
        <TextField
          placeholder="Search countries by name..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          size="small"
          sx={{ minWidth: 300 }}
          fullWidth
        />
      </Box>
      {loading ? (
        <Box sx={{ display: "flex", justifyContent: "center", p: 3 }}>
          <CircularProgress />
        </Box>
      ) : error ? (
        <Alert severity="error">{error}</Alert>
      ) : countries.length === 0 ? (
        <Card>
          <CardContent>
            <Box sx={{ textAlign: "center", p: 3 }}>
              <Typography variant="h6" color="text.secondary">
                No countries found
              </Typography>
              <Button
                variant="contained"
                startIcon={<AddIcon />}
                onClick={() => setShowCreateForm(true)}
                sx={{ mt: 2 }}
              >
                Add First Country
              </Button>
            </Box>
          </CardContent>
        </Card>
      ) : viewMode === "grid" ? (
        <Grid container spacing={3}>
          {countries
            .filter(
              (country) =>
                searchTerm === "" ||
                String(country.name || "")
                  .toLowerCase()
                  .includes(searchTerm.toLowerCase())
            )
            .map((country) => (
              <Grid item xs={12} sm={6} md={4} lg={3} key={country.id}>
                <Card>
                  <CardContent sx={{ pb: 1 }}>
                    <Box sx={{ textAlign: "center", mb: 2 }}>
                      <img
                        src={
                          country.flagUrl ||
                          country.flag ||
                          "/api/placeholder/150/100"
                        }
                        alt={`${country.name} flag`}
                        style={{
                          width: "150px",
                          height: "100px",
                          objectFit: "cover",
                          borderRadius: "8px",
                          border: "1px solid #ddd",
                        }}
                      />
                    </Box>
                    <Typography variant="h6" gutterBottom>
                      {country.name}
                    </Typography>
                    <Box
                      sx={{
                        display: "flex",
                        gap: 1,
                        alignItems: "center",
                        mb: 1,
                      }}
                    >
                      <Chip
                        label={country.isActive ? "Active" : "Inactive"}
                        color={country.isActive ? "success" : "default"}
                        size="small"
                      />
                      <Switch
                        checked={country.isActive}
                        onChange={() => handleToggleActivation(country)}
                        size="small"
                      />
                    </Box>
                  </CardContent>
                  <CardActions
                    sx={{ justifyContent: "space-between", px: 2, pb: 2 }}
                  >
                    <Tooltip title="View Details">
                      <IconButton
                        size="small"
                        color="primary"
                        onClick={() => handleViewCountry(country)}
                      >
                        <ViewIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Edit Country">
                      <IconButton
                        size="small"
                        color="secondary"
                        onClick={() => handleEditCountry(country)}
                      >
                        <EditIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete Country">
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => handleDeleteCountry(country)}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Tooltip>
                  </CardActions>
                </Card>
              </Grid>
            ))}
        </Grid>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Flag</TableCell>
                <TableCell>Name</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {countries
                .filter(
                  (country) =>
                    searchTerm === "" ||
                    String(country.name || "")
                      .toLowerCase()
                      .includes(searchTerm.toLowerCase())
                )
                .map((country) => (
                  <TableRow key={country.id}>
                    <TableCell>
                      <img
                        src={
                          country.flagUrl ||
                          country.flag ||
                          "/api/placeholder/50/30"
                        }
                        alt={`${country.name} flag`}
                        style={{
                          width: "50px",
                          height: "30px",
                          objectFit: "cover",
                          borderRadius: "4px",
                          border: "1px solid #ddd",
                        }}
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">{country.name}</Typography>
                    </TableCell>
                    <TableCell>
                      <Box
                        sx={{ display: "flex", gap: 1, alignItems: "center" }}
                      >
                        <Chip
                          label={country.isActive ? "Active" : "Inactive"}
                          color={country.isActive ? "success" : "default"}
                          size="small"
                        />
                        <Switch
                          checked={country.isActive}
                          onChange={() => handleToggleActivation(country)}
                          size="small"
                        />
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: "flex", gap: 0.5 }}>
                        <Tooltip title="View Details">
                          <IconButton
                            size="small"
                            color="primary"
                            onClick={() => handleViewCountry(country)}
                          >
                            <ViewIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Edit Country">
                          <IconButton
                            size="small"
                            color="secondary"
                            onClick={() => handleEditCountry(country)}
                          >
                            <EditIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Delete Country">
                          <IconButton
                            size="small"
                            color="error"
                            onClick={() => handleDeleteCountry(country)}
                          >
                            <DeleteIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      <Dialog
        open={deleteDialog.open}
        onClose={() => setDeleteDialog({ open: false, country: null })}
      >
        <DialogTitle>Delete Country</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete "{deleteDialog.country?.name}"? This
            action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => setDeleteDialog({ open: false, country: null })}
          >
            Cancel
          </Button>
          <Button
            onClick={confirmDeleteCountry}
            color="error"
            variant="contained"
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

// Users management component
function UsersManagement() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [viewMode, setViewMode] = useState("grid"); // "grid" or "table"
  const [editingUser, setEditingUser] = useState(null);
  const [viewingUser, setViewingUser] = useState(null);
  const [deleteDialog, setDeleteDialog] = useState({ open: false, user: null });
  const [bulkDialog, setBulkDialog] = useState({ open: false });
  const [selectedUsers, setSelectedUsers] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    loadUsers();
  }, []);

  const loadUsers = async () => {
    try {
      setLoading(true);
      const response = await adminAPI.getUsers();
      // API trả về array trực tiếp hoặc object có property content
      setUsers(Array.isArray(response) ? response : response.content || []);
    } catch (err) {
      setError(err.message || "Failed to load users");
    } finally {
      setLoading(false);
    }
  };

  const getInitials = (name) => {
    if (!name) return "U";
    return name
      .split(" ")
      .map((word) => word.charAt(0))
      .join("")
      .toUpperCase()
      .slice(0, 2);
  };

  const handleViewUser = (user) => {
    setViewingUser(user);
  };

  const handleEditUser = (user) => {
    setEditingUser(user);
  };

  const handleDeleteUser = (user) => {
    setDeleteDialog({ open: true, user });
  };

  const confirmDeleteUser = async () => {
    try {
      await adminAPI.deleteUser(deleteDialog.user.id);
      setDeleteDialog({ open: false, user: null });
      loadUsers();
    } catch (err) {
      console.error("Failed to delete user:", err);
    }
  };

  const handleEditSuccess = () => {
    setEditingUser(null);
    loadUsers();
  };

  const handleToggleUserStatus = async (user) => {
    try {
      await adminAPI.updateUserStatus(user.id, !user.enabled);
      loadUsers();
    } catch (err) {
      console.error("Failed to toggle user status:", err);
    }
  };

  const handleSelectUser = (user) => {
    setSelectedUsers((prev) => {
      const isSelected = prev.some((u) => u.id === user.id);
      if (isSelected) {
        return prev.filter((u) => u.id !== user.id);
      } else {
        return [...prev, user];
      }
    });
  };

  const handleBulkDisable = () => {
    if (selectedUsers.length === 0) return;
    setBulkDialog({ open: true });
  };

  const handleBulkSuccess = () => {
    setSelectedUsers([]);
    loadUsers();
  };

  // Filter users based on search term
  const filteredUsers = users.filter(
    (user) =>
      user.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
      user.fullName?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      user.email.toLowerCase().includes(searchTerm.toLowerCase())
  );

  if (viewingUser) {
    return (
      <UserViewForm user={viewingUser} onClose={() => setViewingUser(null)} />
    );
  }

  if (editingUser) {
    return (
      <UserEditForm
        user={editingUser}
        onSuccess={handleEditSuccess}
        onCancel={() => setEditingUser(null)}
      />
    );
  }

  return (
    <Box>
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          mb: 3,
        }}
      >
        <Typography
          variant="h5"
          sx={{ display: "flex", alignItems: "center", gap: 1 }}
        >
          <PersonIcon color="primary" />
          Users Management
        </Typography>
        <Box sx={{ display: "flex", gap: 2, alignItems: "center" }}>
          <Button
            variant={viewMode === "grid" ? "contained" : "outlined"}
            size="small"
            onClick={() => setViewMode("grid")}
          >
            Grid
          </Button>
          <Button
            variant={viewMode === "table" ? "contained" : "outlined"}
            size="small"
            onClick={() => setViewMode("table")}
          >
            Table
          </Button>
        </Box>
      </Box>

      {/* Search and Bulk Operations */}
      <Box
        sx={{
          mb: 3,
          display: "flex",
          gap: 2,
          alignItems: "center",
          flexWrap: "wrap",
        }}
      >
        <TextField
          placeholder="Search users..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          size="small"
          sx={{ minWidth: 200 }}
        />
        {selectedUsers.length > 0 && (
          <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
            <Typography variant="body2" color="text.secondary">
              {selectedUsers.length} selected
            </Typography>
            <Button
              size="small"
              color="error"
              variant="outlined"
              onClick={handleBulkDisable}
              startIcon={<DeleteIcon />}
            >
              Bulk Disable
            </Button>
            <Button size="small" onClick={() => setSelectedUsers([])}>
              Clear
            </Button>
          </Box>
        )}
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {loading ? (
        <Box sx={{ display: "flex", justifyContent: "center", p: 3 }}>
          <CircularProgress />
        </Box>
      ) : viewMode === "grid" ? (
        <Grid container spacing={3}>
          {filteredUsers.map((user) => {
            const isSelected = selectedUsers.some((u) => u.id === user.id);
            return (
              <Grid item xs={12} sm={6} md={4} lg={3} key={user.id}>
                <Card
                  sx={{
                    height: "100%",
                    display: "flex",
                    flexDirection: "column",
                    transition:
                      "transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out",
                    "&:hover": {
                      transform: "translateY(-4px)",
                      boxShadow: (theme) => theme.shadows[8],
                    },
                    border: isSelected ? "2px solid" : "1px solid",
                    borderColor: isSelected ? "primary.main" : "divider",
                  }}
                >
                  <CardHeader
                    avatar={
                      <Avatar
                        src={user.avatarUrl}
                        sx={{
                          width: 60,
                          height: 60,
                          background: user.avatarUrl
                            ? "transparent"
                            : "linear-gradient(135deg, #FFD700 0%, #FFA500 100%)",
                          color: user.avatarUrl ? "transparent" : "#000",
                          fontWeight: 700,
                          fontSize: "1.2rem",
                        }}
                      >
                        {!user.avatarUrl &&
                          getInitials(user.fullName || user.username)}
                      </Avatar>
                    }
                    title={
                      <Typography variant="h6" component="div" noWrap>
                        {user.fullName || user.username}
                      </Typography>
                    }
                    subheader={
                      <Typography variant="body2" color="text.secondary" noWrap>
                        @{user.username}
                      </Typography>
                    }
                    action={null}
                    sx={{ pb: 1 }}
                  />
                  <CardContent sx={{ flexGrow: 1, pt: 0, pb: 2 }}>
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      gutterBottom
                      sx={{ mb: 2 }}
                    >
                      {user.email}
                    </Typography>
                    <Box sx={{ mb: 2 }}>
                      {user.roles?.map((role, index) => (
                        <Chip
                          key={index}
                          label={role}
                          size="small"
                          sx={{ mr: 0.5, mb: 0.5 }}
                          color={role === "ADMIN" ? "error" : "primary"}
                          variant="outlined"
                        />
                      ))}
                    </Box>
                    <Box
                      sx={{
                        display: "flex",
                        gap: 1,
                        alignItems: "center",
                        mb: 2,
                      }}
                    >
                      <Chip
                        label={user.enabled ? "Active" : "Inactive"}
                        color={user.enabled ? "success" : "default"}
                        size="small"
                      />
                      <FormControlLabel
                        control={
                          <Switch
                            checked={user.enabled}
                            onChange={() => handleToggleUserStatus(user)}
                            size="small"
                          />
                        }
                        label=""
                      />
                      <Box
                        sx={{
                          ml: 2,
                          display: "flex",
                          alignItems: "center",
                          gap: 1,
                        }}
                      >
                        <Typography variant="caption" color="text.secondary">
                          Select
                        </Typography>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={isSelected}
                              onChange={() => handleSelectUser(user)}
                              size="small"
                            />
                          }
                          label=""
                        />
                      </Box>
                    </Box>
                  </CardContent>
                  <CardActions
                    sx={{ justifyContent: "center", px: 2, pb: 2, pt: 1 }}
                  >
                    <Tooltip title="View Profile">
                      <IconButton
                        size="small"
                        color="primary"
                        onClick={() => handleViewUser(user)}
                      >
                        <ViewIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Edit User">
                      <IconButton
                        size="small"
                        color="secondary"
                        onClick={() => handleEditUser(user)}
                      >
                        <EditIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete User">
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => handleDeleteUser(user)}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Tooltip>
                  </CardActions>
                </Card>
              </Grid>
            );
          })}
        </Grid>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell padding="checkbox">
                  <Switch
                    checked={
                      selectedUsers.length === filteredUsers.length &&
                      filteredUsers.length > 0
                    }
                    indeterminate={
                      selectedUsers.length > 0 &&
                      selectedUsers.length < filteredUsers.length
                    }
                    onChange={() => {
                      if (selectedUsers.length === filteredUsers.length) {
                        setSelectedUsers([]);
                      } else {
                        setSelectedUsers(filteredUsers);
                      }
                    }}
                  />
                </TableCell>
                <TableCell>Avatar</TableCell>
                <TableCell>User Info</TableCell>
                <TableCell>Email</TableCell>
                <TableCell>Roles</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredUsers.map((user) => {
                const isSelected = selectedUsers.some((u) => u.id === user.id);
                return (
                  <TableRow
                    key={user.id}
                    hover
                    selected={isSelected}
                    sx={{
                      "&.Mui-selected": { backgroundColor: "action.selected" },
                    }}
                  >
                    <TableCell padding="checkbox">
                      <Switch
                        checked={isSelected}
                        onChange={() => handleSelectUser(user)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Avatar
                        src={user.avatarUrl}
                        sx={{
                          width: 50,
                          height: 50,
                          background: user.avatarUrl
                            ? "transparent"
                            : "linear-gradient(135deg, #FFD700 0%, #FFA500 100%)",
                          color: user.avatarUrl ? "transparent" : "#000",
                          fontWeight: 700,
                        }}
                      >
                        {!user.avatarUrl &&
                          getInitials(user.fullName || user.username)}
                      </Avatar>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ py: 0.5 }}>
                        <Typography
                          variant="subtitle2"
                          fontWeight="bold"
                          sx={{ mb: 0.5 }}
                        >
                          {user.fullName || user.username}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          @{user.username}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ py: 0.5 }}>
                        <Typography variant="body2">{user.email}</Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box
                        sx={{
                          display: "flex",
                          flexWrap: "wrap",
                          gap: 0.5,
                          mb: 1,
                        }}
                      >
                        {user.roles?.map((role, index) => (
                          <Chip
                            key={index}
                            label={role}
                            size="small"
                            color={role === "ADMIN" ? "error" : "primary"}
                            variant="outlined"
                          />
                        ))}
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box
                        sx={{
                          display: "flex",
                          gap: 1,
                          alignItems: "center",
                          mb: 1,
                        }}
                      >
                        <Chip
                          label={user.enabled ? "Active" : "Inactive"}
                          color={user.enabled ? "success" : "default"}
                          size="small"
                        />
                        <Switch
                          checked={user.enabled}
                          onChange={() => handleToggleUserStatus(user)}
                          size="small"
                        />
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: "flex", gap: 0.5, py: 0.5 }}>
                        <Tooltip title="View Profile">
                          <IconButton
                            size="small"
                            color="primary"
                            onClick={() => handleViewUser(user)}
                          >
                            <ViewIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Edit User">
                          <IconButton
                            size="small"
                            color="secondary"
                            onClick={() => handleEditUser(user)}
                          >
                            <EditIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Delete User">
                          <IconButton
                            size="small"
                            color="error"
                            onClick={() => handleDeleteUser(user)}
                          >
                            <DeleteIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialog.open}
        onClose={() => setDeleteDialog({ open: false, user: null })}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete user &quot;
            {deleteDialog.user?.username}
            &quot;? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialog({ open: false, user: null })}>
            Cancel
          </Button>
          <Button onClick={confirmDeleteUser} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Bulk Operations Dialog */}
      <BulkOperationsDialog
        open={bulkDialog.open}
        onClose={() => setBulkDialog({ open: false })}
        onSuccess={handleBulkSuccess}
        selectedUsers={selectedUsers}
      />
    </Box>
  );
}

// Dashboard Analytics component
function DashboardAnalytics() {
  const [analytics, setAnalytics] = useState(null);
  const [monthlyStats, setMonthlyStats] = useState([]);
  const currentDate = new Date();
  const [selectedYear, setSelectedYear] = useState(currentDate.getFullYear());
  const [selectedMonth, setSelectedMonth] = useState(
    currentDate.getMonth() + 1
  ); // Current month (1-12)
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    loadAnalytics();
    loadMonthlyStats();
  }, []);

  useEffect(() => {
    loadMonthlyStats();
    // If selected year is current year, set to current month, otherwise set to "all"
    const now = new Date();
    if (selectedYear === now.getFullYear()) {
      setSelectedMonth(now.getMonth() + 1);
    } else {
      setSelectedMonth("all");
    }
  }, [selectedYear]);

  const loadAnalytics = async () => {
    try {
      setLoading(true);

      // Get analytics data from /api/admin/dashboard
      let analyticsData = null;
      try {
        analyticsData = await adminAPI.getAdminStatistics();
        console.log("Analytics API Response:", analyticsData);
      } catch (apiError) {
        console.error("Analytics API Error:", apiError);
        // Create fallback data if API fails
        analyticsData = {
          stats: {
            totalMovies: 0,
            totalUsers: 0,
            totalComments: 0,
            pendingComments: 0,
            activeUsers: 0,
            disabledUsers: 0,
            averageRating: 0,
            totalViews: 0,
            monthlyStats: null,
          },
          pendingComments: [],
          recentUsers: [],
          topMovies: [],
        };
      }

      setAnalytics(analyticsData);
    } catch (err) {
      console.error("General Analytics Error:", err);
      setError(err.message || "Failed to load analytics");
    } finally {
      setLoading(false);
    }
  };

  const loadMonthlyStats = async () => {
    try {
      const response = await adminAPI.getMonthlyStats(selectedYear);
      const stats = Array.isArray(response) ? response : response?.data || [];
      setMonthlyStats(stats);

      // After loading, ensure selected month exists in data
      // If current year and month is selected, check if it exists
      const now = new Date();
      if (
        selectedYear === now.getFullYear() &&
        typeof selectedMonth === "number"
      ) {
        const monthExists = stats.some((m) => m.month === selectedMonth);
        if (!monthExists && stats.length > 0) {
          // If current month doesn't exist, select the first available month
          const firstMonth = stats
            .filter((m) => m.month)
            .map((m) => m.month)
            .sort((a, b) => a - b)[0];
          if (firstMonth) {
            setSelectedMonth(firstMonth);
          } else {
            setSelectedMonth("all");
          }
        } else if (!monthExists) {
          setSelectedMonth("all");
        }
      }
    } catch (err) {
      console.warn("Monthly stats API failed", err);
      setMonthlyStats([]);
    }
  };

  const handleYearChange = (event) => {
    setSelectedYear(Number(event.target.value));
  };

  const handleMonthChange = (event) => {
    setSelectedMonth(event.target.value);
  };

  const getMonthName = (monthNumber) => {
    const months = [
      "January",
      "February",
      "March",
      "April",
      "May",
      "June",
      "July",
      "August",
      "September",
      "October",
      "November",
      "December",
    ];
    return months[monthNumber - 1] || `Month ${monthNumber}`;
  };

  const getMonthOptions = () => {
    const months = monthlyStats
      .filter((m) => m.month)
      .map((m) => m.month)
      .filter((value, index, self) => self.indexOf(value) === index)
      .sort((a, b) => a - b);
    return months;
  };

  const getFilteredMonthlyStats = () => {
    if (selectedMonth === "all") {
      return monthlyStats.filter((m) => m.month);
    }
    return monthlyStats.filter((m) => m.month === Number(selectedMonth));
  };

  const getInitials = (name) => {
    if (!name) return "U";
    return name
      .split(" ")
      .map((word) => word.charAt(0))
      .join("")
      .toUpperCase()
      .slice(0, 2);
  };

  if (loading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography
        variant="h5"
        sx={{ display: "flex", alignItems: "center", gap: 1, mb: 3 }}
      >
        <DashboardIcon color="primary" />
        Dashboard Analytics
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Stats Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ textAlign: "center", p: 2 }}>
            <Typography variant="h4" color="primary" gutterBottom>
              {analytics?.stats?.totalMovies ?? 0}
            </Typography>
            <Typography variant="h6" gutterBottom>
              Total Movies
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {analytics?.stats?.totalComments ?? 0} total comments
            </Typography>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ textAlign: "center", p: 2 }}>
            <Typography variant="h4" color="secondary" gutterBottom>
              {analytics?.stats?.totalUsers ?? 0}
            </Typography>
            <Typography variant="h6" gutterBottom>
              Total Users
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {analytics?.stats?.activeUsers ?? 0} active users
            </Typography>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ textAlign: "center", p: 2 }}>
            <Typography variant="h4" color="success.main" gutterBottom>
              {analytics?.stats?.totalViews ?? 0}
            </Typography>
            <Typography variant="h6" gutterBottom>
              Total Views
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {analytics?.stats?.averageRating ?? 0} avg rating
            </Typography>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ textAlign: "center", p: 2 }}>
            <Typography variant="h4" color="warning.main" gutterBottom>
              {analytics?.stats?.totalComments ?? 0}
            </Typography>
            <Typography variant="h6" gutterBottom>
              Total Comments
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {analytics?.stats?.pendingComments ?? 0} pending
            </Typography>
          </Card>
        </Grid>

        {/* Top Movies */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardHeader
              title={
                <Typography
                  variant="h6"
                  sx={{ display: "flex", alignItems: "center", gap: 1 }}
                >
                  <TrendingIcon color="primary" />
                  Top Movies
                </Typography>
              }
            />
            <CardContent>
              {analytics?.topMovies && analytics.topMovies.length > 0 ? (
                <Stack spacing={2}>
                  {analytics.topMovies.slice(0, 5).map((movie, index) => (
                    <Box
                      key={movie.id || index}
                      sx={{ display: "flex", alignItems: "center", gap: 2 }}
                    >
                      <Typography variant="h6" color="primary">
                        #{index + 1}
                      </Typography>
                      <Avatar
                        src={movie.posterUrl || movie.thumbnailUrl}
                        variant="rounded"
                        sx={{ width: 50, height: 70 }}
                      >
                        {movie.title?.charAt(0) || "M"}
                      </Avatar>
                      <Box sx={{ flexGrow: 1 }}>
                        <Typography variant="subtitle1" fontWeight="bold">
                          {movie.title}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {movie.viewCount || 0} views •{" "}
                          {movie.countryName || ""}
                        </Typography>
                        {movie.categories && movie.categories.length > 0 && (
                          <Box sx={{ display: "flex", gap: 0.5, mt: 0.5 }}>
                            {movie.categories.slice(0, 2).map((cat, idx) => (
                              <Chip
                                key={idx}
                                label={cat}
                                size="small"
                                color="primary"
                                variant="outlined"
                              />
                            ))}
                          </Box>
                        )}
                      </Box>
                      {movie.isFeatured && (
                        <Chip label="Featured" color="primary" size="small" />
                      )}
                      {movie.isTrending && (
                        <Chip label="Trending" color="secondary" size="small" />
                      )}
                    </Box>
                  ))}
                </Stack>
              ) : (
                <Box sx={{ textAlign: "center", py: 4 }}>
                  <Typography color="text.secondary" variant="body1">
                    No top movies available
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Monthly Stats */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardHeader
              title={
                <Box
                  sx={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    flexWrap: "wrap",
                    gap: 1,
                  }}
                >
                  <Typography variant="h6">Monthly Stats</Typography>
                  <Box sx={{ display: "flex", gap: 1 }}>
                    <FormControl size="small" sx={{ minWidth: 100 }}>
                      <InputLabel>Year</InputLabel>
                      <Select
                        value={selectedYear}
                        label="Year"
                        onChange={handleYearChange}
                      >
                        {[0, 1, 2, 3, 4].map((i) => {
                          const y = new Date().getFullYear() - i;
                          return (
                            <MenuItem key={y} value={y}>
                              {y}
                            </MenuItem>
                          );
                        })}
                      </Select>
                    </FormControl>
                    <FormControl size="small" sx={{ minWidth: 120 }}>
                      <InputLabel>Month</InputLabel>
                      <Select
                        value={selectedMonth}
                        label="Month"
                        onChange={handleMonthChange}
                      >
                        <MenuItem value="all">All Months</MenuItem>
                        {getMonthOptions().map((monthNum) => (
                          <MenuItem key={monthNum} value={monthNum}>
                            {getMonthName(monthNum)}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Box>
                </Box>
              }
            />
            <CardContent>
              {monthlyStats && monthlyStats.length > 0 ? (
                <Box
                  sx={{
                    maxHeight: 400,
                    overflowY: "auto",
                    "&::-webkit-scrollbar": {
                      width: 6,
                    },
                    "&::-webkit-scrollbar-track": {
                      background: "rgba(0,0,0,0.1)",
                    },
                    "&::-webkit-scrollbar-thumb": {
                      background: "rgba(0,0,0,0.3)",
                      borderRadius: 3,
                    },
                  }}
                >
                  {getFilteredMonthlyStats().length > 0 ? (
                    <Stack spacing={2}>
                      {getFilteredMonthlyStats().map((monthData, idx) => (
                        <Card key={idx} variant="outlined" sx={{ p: 1.5 }}>
                          <Typography
                            variant="subtitle2"
                            fontWeight="bold"
                            gutterBottom
                            sx={{ mb: 1 }}
                          >
                            {getMonthName(monthData.month)} {monthData.year}
                          </Typography>
                          <Grid container spacing={1}>
                            <Grid item xs={6}>
                              <Typography
                                variant="caption"
                                color="text.secondary"
                              >
                                New Users
                              </Typography>
                              <Typography variant="body2" fontWeight="bold">
                                {monthData.newUsers ?? 0}
                              </Typography>
                            </Grid>
                            <Grid item xs={6}>
                              <Typography
                                variant="caption"
                                color="text.secondary"
                              >
                                New Movies
                              </Typography>
                              <Typography variant="body2" fontWeight="bold">
                                {monthData.newMovies ?? 0}
                              </Typography>
                            </Grid>
                            <Grid item xs={6}>
                              <Typography
                                variant="caption"
                                color="text.secondary"
                              >
                                New Comments
                              </Typography>
                              <Typography variant="body2" fontWeight="bold">
                                {monthData.newComments ?? 0}
                              </Typography>
                            </Grid>
                            <Grid item xs={6}>
                              <Typography
                                variant="caption"
                                color="text.secondary"
                              >
                                Total Views
                              </Typography>
                              <Typography variant="body2" fontWeight="bold">
                                {monthData.totalViews ?? 0}
                              </Typography>
                            </Grid>
                            <Grid item xs={6}>
                              <Typography
                                variant="caption"
                                color="text.secondary"
                              >
                                Revenue
                              </Typography>
                              <Typography variant="body2" fontWeight="bold">
                                ${monthData.revenue ?? 0}
                              </Typography>
                            </Grid>
                          </Grid>
                        </Card>
                      ))}
                    </Stack>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      No data available for selected month
                    </Typography>
                  )}
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No monthly stats available for {selectedYear}
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Users */}
        <Grid item xs={12}>
          <Card>
            <CardHeader
              title={
                <Typography
                  variant="h6"
                  sx={{ display: "flex", alignItems: "center", gap: 1 }}
                >
                  <UsersIcon color="primary" />
                  Recent Users
                </Typography>
              }
            />
            <CardContent>
              {analytics?.recentUsers && analytics.recentUsers.length > 0 ? (
                <Grid container spacing={2}>
                  {analytics.recentUsers.slice(0, 3).map((user) => (
                    <Grid item xs={12} sm={6} md={4} key={user.id}>
                      <Card variant="outlined">
                        <CardContent>
                          <Box
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              gap: 2,
                            }}
                          >
                            <Avatar
                              src={user.avatarUrl}
                              sx={{
                                width: 50,
                                height: 50,
                                background: user.avatarUrl
                                  ? "transparent"
                                  : "linear-gradient(135deg, #FFD700 0%, #FFA500 100%)",
                                color: user.avatarUrl ? "transparent" : "#000",
                                fontWeight: 700,
                              }}
                            >
                              {!user.avatarUrl &&
                                getInitials(user.fullName || user.username)}
                            </Avatar>
                            <Box sx={{ flexGrow: 1 }}>
                              <Typography variant="subtitle1" fontWeight="bold">
                                {user.fullName || user.username}
                              </Typography>
                              <Typography
                                variant="body2"
                                color="text.secondary"
                              >
                                @{user.username}
                              </Typography>
                              <Typography
                                variant="caption"
                                color="text.secondary"
                              >
                                {user.email}
                              </Typography>
                              <Box sx={{ display: "flex", gap: 0.5, mt: 0.5 }}>
                                {user.roles?.map((role, idx) => (
                                  <Chip
                                    key={idx}
                                    label={role}
                                    size="small"
                                    color={
                                      role === "ADMIN" ? "error" : "primary"
                                    }
                                    variant="outlined"
                                  />
                                ))}
                                <Chip
                                  label={user.enabled ? "Active" : "Inactive"}
                                  size="small"
                                  color={user.enabled ? "success" : "default"}
                                />
                              </Box>
                            </Box>
                          </Box>
                          <Box
                            sx={{
                              display: "flex",
                              gap: 2,
                              mt: 2,
                              pt: 2,
                              borderTop: 1,
                              borderColor: "divider",
                            }}
                          >
                            <Box>
                              <Typography
                                variant="caption"
                                color="text.secondary"
                              >
                                Comments
                              </Typography>
                              <Typography variant="body2" fontWeight="bold">
                                {user.totalComments ?? 0}
                              </Typography>
                            </Box>
                            <Box>
                              <Typography
                                variant="caption"
                                color="text.secondary"
                              >
                                Movies
                              </Typography>
                              <Typography variant="body2" fontWeight="bold">
                                {user.totalMovies ?? 0}
                              </Typography>
                            </Box>
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              ) : (
                <Box sx={{ textAlign: "center", py: 4 }}>
                  <Typography color="text.secondary" variant="body1">
                    No recent users available
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

// Reports Management component
function ReportsManagement() {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    loadReports();
  }, []);

  const loadReports = async () => {
    try {
      setLoading(true);
      const response = await adminAPI.getReports();
      // API trả về array trực tiếp hoặc object có property content
      setReports(Array.isArray(response) ? response : response.content || []);
    } catch (err) {
      setError(err.message || "Failed to load reports");
    } finally {
      setLoading(false);
    }
  };

  const handleResolveReport = async (reportId) => {
    try {
      await adminAPI.resolveReport(reportId);
      loadReports();
    } catch (err) {
      console.error("Failed to resolve report:", err);
    }
  };

  return (
    <Box>
      <Typography
        variant="h5"
        sx={{ display: "flex", alignItems: "center", gap: 1, mb: 3 }}
      >
        <TrendingIcon color="primary" />
        Reports Management
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Search */}
      <Box sx={{ mb: 3 }}>
        <TextField
          placeholder="Search reports by type, description, or reporter..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          size="small"
          sx={{ minWidth: 300 }}
          fullWidth
        />
      </Box>

      {loading ? (
        <Box sx={{ display: "flex", justifyContent: "center", p: 3 }}>
          <CircularProgress />
        </Box>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>ID</TableCell>
                <TableCell>Type</TableCell>
                <TableCell>Description</TableCell>
                <TableCell>Reporter</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Date</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {reports
                .filter(
                  (report) =>
                    searchTerm === "" ||
                    String(report.reportType || "")
                      .toLowerCase()
                      .includes(searchTerm.toLowerCase()) ||
                    String(report.reason || "")
                      .toLowerCase()
                      .includes(searchTerm.toLowerCase()) ||
                    String(report.reporter?.username || "")
                      .toLowerCase()
                      .includes(searchTerm.toLowerCase())
                )
                .map((report) => (
                  <TableRow key={report.id} hover>
                    <TableCell>
                      <Typography variant="body2" fontWeight="bold">
                        #{report.id}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={report.reportType || "Unknown"}
                        size="small"
                        color={
                          report.reportType === "SPAM" ? "error" : "warning"
                        }
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" noWrap sx={{ maxWidth: 200 }}>
                        {report.reason || "No description"}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {report.reporter?.username || "Unknown"}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={
                          report.status === "RESOLVED" ? "Resolved" : "Pending"
                        }
                        color={
                          report.status === "RESOLVED" ? "success" : "default"
                        }
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {new Date(report.createdAt).toLocaleDateString()}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      {report.status !== "RESOLVED" && (
                        <Button
                          size="small"
                          variant="contained"
                          color="success"
                          onClick={() => handleResolveReport(report.id)}
                        >
                          Resolve
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
}

// Main Admin Management component
export default function AdminManagement() {
  const navigate = useNavigate();
  const [tabValue, setTabValue] = useState(0);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleBackToHome = () => {
    navigate("/");
  };

  return (
    <AdminRoute>
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
          <Button
            variant="outlined"
            startIcon={<ArrowBackIcon />}
            onClick={handleBackToHome}
            sx={{
              borderColor: "primary.main",
              color: "primary.main",
              "&:hover": {
                backgroundColor: "primary.main",
                color: "white",
              },
            }}
          >
            Back to Home
          </Button>
          <Typography variant="h4" sx={{ flexGrow: 1 }}>
            Admin Management
          </Typography>
        </Box>

        <Box sx={{ borderBottom: 1, borderColor: "divider" }}>
          <Tabs value={tabValue} onChange={handleTabChange}>
            <Tab
              icon={<MoviesIcon />}
              label="Movies"
              iconPosition="start"
              sx={{
                minHeight: 48,
                "&.Mui-selected": {
                  color: "primary.main",
                },
              }}
            />
            <Tab
              icon={<UsersIcon />}
              label="Users"
              iconPosition="start"
              sx={{
                minHeight: 48,
                "&.Mui-selected": {
                  color: "primary.main",
                },
              }}
            />
            <Tab
              icon={<PublicIcon />}
              label="Countries"
              iconPosition="start"
              sx={{
                minHeight: 48,
                "&.Mui-selected": {
                  color: "primary.main",
                },
              }}
            />
            <Tab
              icon={<CategoryIcon />}
              label="Categories"
              iconPosition="start"
              sx={{
                minHeight: 48,
                "&.Mui-selected": {
                  color: "primary.main",
                },
              }}
            />
            <Tab
              icon={<TheaterComedyIcon />}
              label="Actors"
              iconPosition="start"
              sx={{
                minHeight: 48,
                "&.Mui-selected": {
                  color: "primary.main",
                },
              }}
            />
            <Tab
              icon={<PersonIcon />}
              label="Directors"
              iconPosition="start"
              sx={{
                minHeight: 48,
                "&.Mui-selected": {
                  color: "primary.main",
                },
              }}
            />
            <Tab
              icon={<StatisticsIcon />}
              label="Statistics"
              iconPosition="start"
              sx={{
                minHeight: 48,
                "&.Mui-selected": {
                  color: "primary.main",
                },
              }}
            />
            <Tab
              icon={<ReportsIcon />}
              label="Reports"
              iconPosition="start"
              sx={{
                minHeight: 48,
                "&.Mui-selected": {
                  color: "primary.main",
                },
              }}
            />
          </Tabs>
        </Box>

        <TabPanel value={tabValue} index={0}>
          <MoviesManagement />
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <UsersManagement />
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <CountriesManagement />
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          <CategoriesManagement />
        </TabPanel>

        <TabPanel value={tabValue} index={4}>
          <ActorsManagement />
        </TabPanel>

        <TabPanel value={tabValue} index={5}>
          <DirectorsManagement />
        </TabPanel>

        <TabPanel value={tabValue} index={6}>
          <DashboardAnalytics />
        </TabPanel>

        <TabPanel value={tabValue} index={7}>
          <ReportsManagement />
        </TabPanel>
      </Container>
    </AdminRoute>
  );
}
