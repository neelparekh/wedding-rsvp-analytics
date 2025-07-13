import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MultiLabelBinarizer
from google.oauth2 import service_account
from googleapiclient.discovery import build
import io

# Color scheme constants
COLOR_BRIDE = '#FF69B4'
COLOR_GROOM = '#4169E1'
COLOR_RSVP_YES = '#32CD32'
COLOR_RSVP_NO = '#FF6B6B'
COLOR_RSVP_UNANS = '#FFA500'

# Events to track
EVENTS = ['Haldi', 'Mehendi', 'Sangeet', 'Wedding', 'Reception']

st.set_page_config(
    page_title="Wedding Guest Analytics",
    page_icon="💒",
    layout="wide"
)

def check_password():
    """Password protection for the app"""
    def password_entered():
        # Get password from Streamlit secrets
        correct_password = st.secrets.get("app_password")
        
        if st.session_state["password"] == correct_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    # Check if password is already correct
    if "password_correct" not in st.session_state:
        st.title("🔒 Wedding Guest Analytics")
        st.markdown("Please enter the password to access the dashboard")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.title("🔒 Wedding Guest Analytics")
        st.markdown("Please enter the password to access the dashboard")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("❌ Incorrect password. Please try again.")
        return False
    else:
        return True

@st.cache_resource
def init_google_drive():
    """Initialize Google Drive service"""
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        return build('drive', 'v3', credentials=credentials)
    except Exception as e:
        st.error(f"❌ Error connecting to Google Drive: {e}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_latest_csv_from_folder(folder_id):
    """Get the latest CSV file from a Google Drive folder"""
    try:
        drive_service = init_google_drive()
        if drive_service is None:
            return None, None
        
        # Search for CSV files in the folder
        query = f"'{folder_id}' in parents and mimeType='text/csv' and trashed=false"
        results = drive_service.files().list(
            q=query,
            orderBy='modifiedTime desc',  # Latest first
            fields="files(id,name,modifiedTime)",
            pageSize=10
        ).execute()
        
        files = results.get('files', [])
        
        if not files:
            st.error("❌ No CSV files found in the specified folder")
            return None, None
        
        # Return the latest file
        latest_file = files[0]
        return latest_file['id'], latest_file
        
    except Exception as e:
        st.error(f"❌ Error accessing folder: {e}")
        return None, None

@st.cache_data(ttl=300)
def load_csv_from_drive(file_id):
    """Load CSV from Google Drive"""
    try:
        drive_service = init_google_drive()
        if drive_service is None:
            return None
            
        # Download file content
        request = drive_service.files().get_media(fileId=file_id)
        file_content = request.execute()
        
        # Convert to DataFrame
        csv_string = file_content.decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_string))
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading file from Google Drive: {e}")
        return None

@st.cache_data(ttl=300)
def load_and_process_data(uploaded_file=None, use_drive=False, file_id=None, folder_id=None):
    """Load CSV and create invitation indicator columns"""
    try:
        if use_drive and folder_id:
            # Load latest CSV from folder
            file_id, file_info = get_latest_csv_from_folder(folder_id)
            if file_id is None:
                return None, None
            df = load_csv_from_drive(file_id)
            if df is None:
                return None, None
        elif use_drive and file_id:
            # Load from specific Google Drive file
            df = load_csv_from_drive(file_id)
            file_info = None
            if df is None:
                return None, None
        elif uploaded_file is not None:
            # Load from uploaded file
            df = pd.read_csv(uploaded_file)
            file_info = None
        else:
            # Fallback to local file for development
            try:
                df = pd.read_csv('full-guest-list-Jul12.csv')
                file_info = None
            except FileNotFoundError:
                return None, None
        
        # Expected columns
        keep_cols = ['first name', 'last name', 'party', 'tags', 'mehendi rsvp', 'sangeet rsvp', 'wedding rsvp', 'reception rsvp', 'haldi rsvp']
        
        # Check if required columns exist
        missing_cols = [col for col in keep_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            return None, None
        
        df = df[keep_cols]

        # Convert tags to invitation indicators
        tags_split = df['tags'].fillna('').str.split(',').apply(
            lambda x: [tag.strip() + ' Invitation' for tag in x if tag.strip()]
        )
        
        mlb = MultiLabelBinarizer()
        tag_indicators = mlb.fit_transform(tags_split)
        tag_df = pd.DataFrame(tag_indicators, columns=mlb.classes_, index=df.index)
        
        processed_df = pd.concat([df, tag_df], axis=1)
        return processed_df, file_info
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        return None, None

def get_event_stats(df):
    """Calculate comprehensive statistics for each event"""
    stats = []
    
    for event in EVENTS:
        b_col = f'B {event} Invitation'
        g_col = f'G {event} Invitation'
        rsvp_col = f'{event.lower()} rsvp'
        
        # Count invitations by side
        b_invites = df[b_col].sum() if b_col in df.columns else 0
        g_invites = df[g_col].sum() if g_col in df.columns else 0
        total_invites = b_invites + g_invites
        
        # Count RSVP responses
        total_rsvps = df[rsvp_col].notna().sum() if rsvp_col in df.columns else 0
        yes_rsvps = (df[rsvp_col].str.lower() == 'attending').sum() if rsvp_col in df.columns else 0
        no_rsvps = (df[rsvp_col].str.lower() == 'not attending').sum() if rsvp_col in df.columns else 0
        
        # Count yes RSVPs by side
        bride_yes = ((df[rsvp_col].str.lower() == 'attending') & (df[b_col] == 1)).sum() if rsvp_col in df.columns and b_col in df.columns else 0
        groom_yes = ((df[rsvp_col].str.lower() == 'attending') & (df[g_col] == 1)).sum() if rsvp_col in df.columns and g_col in df.columns else 0
        
        # Count unanswered invitations
        if all(col in df.columns for col in [b_col, g_col, rsvp_col]):
            has_invite = (df[b_col] == 1) | (df[g_col] == 1)
            unanswered = (has_invite & df[rsvp_col].isna()).sum()
        else:
            unanswered = 0
        
        stats.append({
            'Event': event,
            'Bride Invites': b_invites,
            'Groom Invites': g_invites,
            'Total Invites': total_invites,
            'Total RSVPs': total_rsvps,
            'Yes RSVPs': yes_rsvps,
            'No RSVPs': no_rsvps,
            'Bride Yes RSVPs': bride_yes,
            'Groom Yes RSVPs': groom_yes,
            'Unanswered': unanswered,
            'Response Rate': f"{(total_rsvps/total_invites*100):.1f}%" if total_invites > 0 else "0%"
        })
    
    return pd.DataFrame(stats)

def get_unanswered_guests(df, event):
    """Get list of guests with unanswered invitations for specific event"""
    b_col = f'B {event} Invitation'
    g_col = f'G {event} Invitation'
    rsvp_col = f'{event.lower()} rsvp'
    
    if not all(col in df.columns for col in [b_col, g_col, rsvp_col]):
        return pd.DataFrame()
    
    # Find guests with invitations but no RSVP
    has_invite = (df[b_col] == 1) | (df[g_col] == 1)
    unanswered_mask = has_invite & df[rsvp_col].isna()
    
    if not unanswered_mask.any():
        return pd.DataFrame()
    
    unanswered_df = df[unanswered_mask].copy()
    unanswered_df['Side'] = unanswered_df.apply(
        lambda row: 'Bride' if row[b_col] == 1 else 'Groom', axis=1
    )
    
    return unanswered_df[['first name', 'last name', 'party', 'Side']].reset_index(drop=True)

def create_pie_chart(values, names, title, color_map):
    """Create pie chart with consistent colors"""
    # Filter out zero values for cleaner charts
    filtered_data = [(v, n) for v, n in zip(values, names) if v > 0]
    if not filtered_data:
        return None
    
    chart_values, chart_names = zip(*filtered_data)
    
    fig = px.pie(
        values=chart_values,
        names=chart_names,
        title=title,
        color=chart_names,
        color_discrete_map=color_map
    )
    return fig

def main():
    # Check password before showing the app
    if not check_password():
        return
        
    st.title("💒 Wedding Guest Analytics Dashboard")
    st.markdown("---")
    
    # Data source selection
    st.subheader("📊 Choose Data Source")
    data_source = st.radio(
        "How would you like to load your guest list?",
        ["📂 Google Drive Folder (Latest CSV)", "📄 Specific Google Drive File", "📁 Upload CSV File"],
        help="Folder option automatically uses the newest CSV file in your folder"
    )
    
    df = None
    file_info = None
    
    if data_source.startswith("📂") or data_source.startswith("📄"):  # Google Drive options
        st.subheader("📂 Google Drive Configuration")
        
        # Check if Google Drive is configured
        if "gcp_service_account" not in st.secrets:
            st.error("❌ Google Drive not configured. Please add your service account credentials to secrets.")
            st.markdown("""
            **To set up Google Drive:**
            1. Create a service account in Google Cloud Console
            2. Download the JSON key file
            3. Add the JSON content to your Streamlit secrets as `gcp_service_account`
            4. Share your CSV file/folder with the service account email
            """)
            st.stop()
        
        if data_source.startswith("📂"):  # Folder option
            # Folder ID input
            # folder_id = st.text_input(
            #     "Google Drive Folder ID",
            #     help="Copy the folder ID from your Google Drive URL: https://drive.google.com/drive/folders/FOLDER_ID_HERE",
            #     placeholder="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"
            # )

            folder_id="1IQBNaL_37ucZpTZkBdivW9oe6w-_Xgig"
            
            if folder_id:
                with st.spinner("Finding latest CSV in folder..."):
                    result = load_and_process_data(use_drive=True, folder_id=folder_id)
                    if result[0] is not None:
                        df, file_info = result
                        st.success(f"✅ Loaded {len(df)} guests from latest CSV in folder!")
                        
                        # Show file info
                        if file_info:
                            st.info(f"📄 Latest file: **{file_info.get('name', 'Unknown')}** | Modified: {file_info.get('modifiedTime', 'Unknown')}")
                    else:
                        st.error("❌ Could not load latest CSV from folder. Please check the folder ID and permissions.")
                        st.stop()
            else:
                st.info("👆 Please enter your Google Drive folder ID to continue")
                with st.expander("🔍 How to find Folder ID"):
                    st.markdown("""
                    1. Open your folder in Google Drive
                    2. Look at the URL: `https://drive.google.com/drive/folders/FOLDER_ID_HERE`
                    3. Copy the **FOLDER_ID_HERE** part
                    4. Make sure your folder is shared with the service account email
                    """)
                st.stop()
                
        else:  # Specific file option
            # File ID input
            file_id = st.text_input(
                "Google Drive File ID",
                help="Copy the file ID from your Google Drive URL: https://drive.google.com/file/d/FILE_ID_HERE/view",
                placeholder="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"
            )
            
            if file_id:
                with st.spinner("Loading data from Google Drive..."):
                    result = load_and_process_data(use_drive=True, file_id=file_id)
                    if result[0] is not None:
                        df, _ = result
                        st.success(f"✅ Loaded {len(df)} guests from Google Drive!")
                        
                        # Show last update info
                        drive_service = init_google_drive()
                        if drive_service:
                            try:
                                file_info_api = drive_service.files().get(fileId=file_id, fields="modifiedTime,name").execute()
                                st.info(f"📄 File: {file_info_api.get('name', 'Unknown')} | Last modified: {file_info_api.get('modifiedTime', 'Unknown')}")
                            except:
                                pass
                    else:
                        st.error("❌ Could not load file from Google Drive. Please check the file ID and permissions.")
                        st.stop()
            else:
                st.info("👆 Please enter your Google Drive file ID to continue")
                st.stop()
            
    else:  # Upload CSV File
        st.subheader("📁 Upload Your Guest List")
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type="csv",
            help="Upload your guest list CSV with the required columns"
        )
        
        # Show expected format
        with st.expander("📋 Expected CSV Format"):
            st.markdown("""
            Your CSV should have these columns:
            - **first name**: Guest's first name
            - **last name**: Guest's last name  
            - **party**: Party/group information
            - **tags**: Comma-separated tags (e.g., "B Wedding, B Reception, Save the Date")
            - **mehendi rsvp**: RSVP status ("attending", "not attending", or empty)
            - **sangeet rsvp**: RSVP status ("attending", "not attending", or empty)
            - **wedding rsvp**: RSVP status ("attending", "not attending", or empty)
            - **reception rsvp**: RSVP status ("attending", "not attending", or empty)
            - **haldi rsvp**: RSVP status ("attending", "not attending", or empty)
            """)
        
        if uploaded_file is not None:
            result = load_and_process_data(uploaded_file=uploaded_file)
            if result[0] is not None:
                df, _ = result
                st.success(f"✅ Loaded {len(df)} guests from uploaded file!")
            else:
                st.error("❌ Error processing the uploaded file. Please check the format and try again.")
                st.stop()
        else:
            st.info("👆 Please upload your guest list CSV file to begin analysis")
            st.stop()
    
    # Only proceed if we have valid data
    if df is None:
        st.stop()
        
    stats_df = get_event_stats(df)
    
    # Create navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Event Analytics", "📋 Unanswered Invitations", "📄 Raw Data"])
    
    with tab1:
        st.header("Wedding Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Guests", len(df))
        with col2:
            st.metric("Total Invitations", stats_df['Total Invites'].sum())
        with col3:
            st.metric("Total RSVPs", stats_df['Total RSVPs'].sum())
        with col4:
            total_invites = stats_df['Total Invites'].sum()
            overall_response_rate = (stats_df['Total RSVPs'].sum() / total_invites * 100) if total_invites > 0 else 0
            st.metric("Overall Response Rate", f"{overall_response_rate:.1f}%")

        # Event attendee counts
        st.subheader("Event Summary")
        cols = st.columns(len(EVENTS))
        for event, col in zip(EVENTS, cols):
            attendees = stats_df.loc[stats_df['Event'] == event, 'Yes RSVPs'].iloc[0]
            with col:
                st.metric(f"{event} Attendees", attendees)
        
        st.dataframe(stats_df, use_container_width=True)
        
        # Overview charts
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Invitations by Event', 'RSVPs by Event'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Invitations by side
        fig.add_trace(go.Bar(name='Bride Side', x=stats_df['Event'], y=stats_df['Bride Invites'], marker_color=COLOR_BRIDE), row=1, col=1)
        fig.add_trace(go.Bar(name='Groom Side', x=stats_df['Event'], y=stats_df['Groom Invites'], marker_color=COLOR_GROOM), row=1, col=1)
        
        # RSVP status
        fig.add_trace(go.Bar(name='Attending', x=stats_df['Event'], y=stats_df['Yes RSVPs'], marker_color=COLOR_RSVP_YES, showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(name='Not Attending', x=stats_df['Event'], y=stats_df['No RSVPs'], marker_color=COLOR_RSVP_NO, showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(name='Unanswered', x=stats_df['Event'], y=stats_df['Unanswered'], marker_color=COLOR_RSVP_UNANS, showlegend=False), row=1, col=2)
        
        fig.update_layout(height=500, title_text="Wedding Guest Analytics Overview")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Per-Event Analytics")
        
        selected_event = st.selectbox("Select Event", EVENTS)
        event_stats = stats_df[stats_df['Event'] == selected_event].iloc[0]
        
        # Event metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{selected_event} - Total Invites", event_stats['Total Invites'])
        with col2:
            st.metric(f"{selected_event} - Response Rate", event_stats['Response Rate'])
        with col3:
            st.metric(f"{selected_event} - Total Attendees", event_stats['Yes RSVPs'])
        
        # Event breakdown charts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Invitation breakdown by side
            fig_invite = create_pie_chart(
                [event_stats['Bride Invites'], event_stats['Groom Invites']],
                ['Bride Side', 'Groom Side'],
                f"{selected_event} - Invitation Breakdown",
                {'Bride Side': COLOR_BRIDE, 'Groom Side': COLOR_GROOM}
            )
            if fig_invite:
                st.plotly_chart(fig_invite, use_container_width=True)
            else:
                st.info(f"No invitations for {selected_event}")
        
        with col2:
            # RSVP status breakdown
            fig_rsvp = create_pie_chart(
                [event_stats['Yes RSVPs'], event_stats['No RSVPs'], event_stats['Unanswered']],
                ['Attending', 'Not Attending', 'Unanswered'],
                f"{selected_event} - RSVP Status",
                {'Attending': COLOR_RSVP_YES, 'Not Attending': COLOR_RSVP_NO, 'Unanswered': COLOR_RSVP_UNANS}
            )
            if fig_rsvp:
                st.plotly_chart(fig_rsvp, use_container_width=True)
            else:
                st.info(f"No RSVP data for {selected_event}")

        with col3:
            # Attendees by side
            fig_attendees = create_pie_chart(
                [event_stats['Bride Yes RSVPs'], event_stats['Groom Yes RSVPs']],
                ['Bride Attendees', 'Groom Attendees'],
                f"{selected_event} - Attendees by Side",
                {'Bride Attendees': COLOR_BRIDE, 'Groom Attendees': COLOR_GROOM}
            )
            if fig_attendees:
                st.plotly_chart(fig_attendees, use_container_width=True)
            else:
                st.info(f"No attendees yet for {selected_event}")
    
    with tab3:
        st.header("📋 Unanswered Invitations")
        st.markdown("Guests who have been invited but haven't responded yet")
        
        selected_event_unans = st.selectbox("Select Event for Unanswered List", EVENTS, key="unanswered_event")
        unanswered_df = get_unanswered_guests(df, selected_event_unans)
        
        if len(unanswered_df) > 0:
            st.subheader(f"Unanswered {selected_event_unans} Invitations ({len(unanswered_df)} guests)")
            
            # Summary metrics
            side_summary = unanswered_df['Side'].value_counts()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Unanswered", len(unanswered_df))
            with col2:
                st.metric("Bride Side", side_summary.get('Bride', 0))
            with col3:
                st.metric("Groom Side", side_summary.get('Groom', 0))
            
            # Filter and display
            side_filter = st.multiselect("Filter by Side", ['Bride', 'Groom'], default=['Bride', 'Groom'])
            filtered_df = unanswered_df[unanswered_df['Side'].isin(side_filter)]
            
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download functionality
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label=f"Download {selected_event_unans} Unanswered List",
                data=csv,
                file_name=f"unanswered_{selected_event_unans.lower()}_invitations.csv",
                mime="text/csv"
            )
        else:
            st.success(f"🎉 All {selected_event_unans} invitations have been answered!")
    
    with tab4:
        st.header("📄 Raw Data")
        st.markdown("Full dataset with all processed columns")
        
        # Show invitation columns created
        invitation_cols = [col for col in df.columns if 'Invitation' in col]
        st.write(f"**Invitation columns created:** {len(invitation_cols)}")
        st.write(invitation_cols)
        
        # Display and download full dataset
        st.dataframe(df, use_container_width=True)
        
        csv_full = df.to_csv(index=False)
        st.download_button(
            label="Download Full Dataset",
            data=csv_full,
            file_name="wedding_guest_analytics_full.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()