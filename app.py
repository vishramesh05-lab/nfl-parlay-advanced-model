import io, zipfile, chardet

def safe_read_file(file_like):
    """Reads CSV, ZIP, or Excel; detects encoding automatically and never crashes."""
    try:
        # Read a small sample to guess encoding
        sample = file_like.read(4096)
        file_like.seek(0)
        guess = chardet.detect(sample).get("encoding", "utf-8")

        # Try UTF-8 or guessed encoding
        try:
            return pd.read_csv(file_like, encoding=guess)
        except pd.errors.EmptyDataError:
            st.error("❌ The uploaded file is empty.")
            return pd.DataFrame()
        except pd.errors.ParserError:
            # If file is ZIP → open first CSV inside
            file_like.seek(0)
            if zipfile.is_zipfile(file_like):
                with zipfile.ZipFile(file_like, "r") as z:
                    csv_files = [n for n in z.namelist() if n.endswith(".csv")]
                    if not csv_files:
                        st.error("⚠️ ZIP found but no CSV inside.")
                        return pd.DataFrame()
                    csv_name = csv_files[0]
                    st.info(f"📦 Extracting {csv_name} from ZIP...")
                    with z.open(csv_name) as csv_file:
                        sample2 = csv_file.read(4096)
                        enc2 = chardet.detect(sample2).get("encoding", "utf-8")
                        csv_file = io.BytesIO(sample2 + csv_file.read())  # reload whole file
                        csv_file.seek(0)
                        return pd.read_csv(csv_file, encoding=enc2)
            # Try Excel fallback
            file_like.seek(0)
            try:
                return pd.read_excel(file_like)
            except Exception:
                st.error("⚠️ Could not parse as CSV, ZIP, or Excel.")
                return pd.DataFrame()
        except UnicodeDecodeError:
            # Last resort re-detect encoding
            file_like.seek(0)
            raw = file_like.read()
            enc3 = chardet.detect(raw).get("encoding", "latin1")
            file_like.seek(0)
            return pd.read_csv(file_like, encoding=enc3)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_data(uploaded_file=None):
    """Loads dataset from upload or from default file path."""
    try:
        if uploaded_file is not None:
            df = safe_read_file(uploaded_file)
        else:
            with open(DATA_FILE, "rb") as f:
                df = safe_read_file(f)
        if df.empty:
            st.error("⚠️ No data found in file.")
            return pd.DataFrame()
        df.columns = [c.lower().strip() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()
